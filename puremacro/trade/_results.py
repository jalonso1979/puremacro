"""Frozen-dataclass result containers for puremacro.trade."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence, Tuple

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst


@dataclass(frozen=True)
class TradeCalibrationResult:
    """Calibrated parameters and baseline endowments for the CGE trade model.

    Stores the output of :func:`calibrate_trade_model`, encapsulating Input-Output
    technical coefficients, final demand sourcing coefficients, Cobb-Douglas
    technology parameters, factor endowments, production taxes, net foreign
    transfers, and baseline gross outputs.

    Attributes
    ----------
    a : np.ndarray, shape (847, 11, 77) or (ns*nc, ns, nc)
        Bilateral intermediate input-output technical coefficients:
        ``a[i, j, k] = intermediate_use / gross_output``.
    afd : np.ndarray, shape (847, 3, 77) or (ns*nc, nfd, nc)
        Bilateral final demand expenditure coefficients:
        ``afd[i, ifd, k] = final_use / total_final_use``.
    alpha : np.ndarray, shape (1, 11, 77) or (1, ns, nc)
        Capital share in value added (Cobb-Douglas exponent, uniform 1/3).
    beta : np.ndarray, shape (1, 11, 77) or (1, ns, nc)
        Total factor productivity scale parameter in value-added production.
    k_endow : np.ndarray, shape (1, 77) or (1, nc)
        Total national capital endowments by country (also accessible as ``KT``).
    l_endow : np.ndarray, shape (1, 77) or (1, nc)
        Total national labor endowments by country (also accessible as ``LT``).
    invforT : np.ndarray, shape (1, 77) or (1, nc)
        Net foreign transfers / trade surplus baseline (current account balance).
    tax : np.ndarray, shape (1, 11, 77) or (1, ns, nc)
        Baseline net production tax rates.
    ytot : np.ndarray, shape (1, 11, 77) or (1, ns, nc)
        Baseline gross output quantities (also accessible as ``ytot_base``).
    theta : np.ndarray | None, shape (1, 3, 77) or (1, nfd, nc), default None
        Final demand expenditure shares across consumption categories.
    tax_fd : np.ndarray | None, shape (1, 3, 77) or (1, nfd, nc), default None
        Final demand tax rates across categories.
    TT : np.ndarray | None, shape (1, 77) or (1, nc), default None
        Intermediate production taxes aggregated by country.
    TTfd : np.ndarray | None, shape (1, 77) or (1, nc), default None
        Final demand taxes aggregated by country.
    T : np.ndarray | None, shape (1, 1, 77) or (1, 1, nc), default None
        Total government tax transfers in baseline level.
    a_3d : np.ndarray | None, shape (ns*nc, ns, nc), default None
        3D intermediate technical coefficients (identical to ``a`` when 3D).
    afd_3d : np.ndarray | None, shape (ns*nc, nfd, nc), default None
        3D final demand sourcing coefficients (identical to ``afd`` when 3D).
    a_4d : np.ndarray | None, shape (ns, nc, ns, nc), default None
        4D intermediate technical coefficient tensor.
    afd_4d : np.ndarray | None, shape (ns, nc, nfd, nc), default None
        4D final demand sourcing coefficient tensor.
    data_calibra : np.ndarray | None, shape (850, 1078), default None
        Reconstructed benchmark transaction matrix matching original data.
    n_countries : int, default 77
        Number of countries in the calibrated system.
    n_sectors : int, default 11
        Number of industrial sectors in the calibrated system.
    n_final_demand : int, default 3
        Number of final demand categories.
    country_codes : tuple[str, ...], default ()
        ISO-3 country identifier codes.
    sector_codes : tuple[str, ...], default ()
        Sector identifier codes.
    metadata : dict[str, Any], default empty
        Optional provenance and calibration metadata.
    """

    a: np.ndarray
    afd: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    k_endow: np.ndarray
    l_endow: np.ndarray
    invforT: np.ndarray
    tax: np.ndarray
    ytot: np.ndarray
    theta: np.ndarray | None = None
    tax_fd: np.ndarray | None = None
    TT: np.ndarray | None = None
    TTfd: np.ndarray | None = None
    T: np.ndarray | None = None
    a_3d: np.ndarray | None = None
    afd_3d: np.ndarray | None = None
    a_4d: np.ndarray | None = None
    afd_4d: np.ndarray | None = None
    data_calibra: np.ndarray | None = None
    n_countries: int = 77
    n_sectors: int = 11
    n_final_demand: int = 3
    country_codes: tuple[str, ...] = field(default_factory=tuple)
    sector_codes: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate presence, types, and non-emptiness of all required arrays."""
        required = {
            "a": self.a,
            "afd": self.afd,
            "alpha": self.alpha,
            "beta": self.beta,
            "k_endow": self.k_endow,
            "l_endow": self.l_endow,
            "invforT": self.invforT,
            "tax": self.tax,
            "ytot": self.ytot,
        }
        for name, arr in required.items():
            if arr is None:
                raise ValueError(
                    f"TradeCalibrationResult requires array '{name}' (cannot be None)."
                )
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"TradeCalibrationResult field '{name}' must be a numpy.ndarray, "
                    f"got {type(arr).__name__}."
                )
            if arr.size == 0:
                raise ValueError(
                    f"TradeCalibrationResult field '{name}' cannot be empty (size is 0)."
                )

        # Normalize country_codes and sector_codes to tuples
        if not isinstance(self.country_codes, tuple):
            object.__setattr__(self, "country_codes", tuple(self.country_codes))
        if not isinstance(self.sector_codes, tuple):
            object.__setattr__(self, "sector_codes", tuple(self.sector_codes))

        # Synchronize 3D representations
        if self.a_3d is None and self.a.ndim == 3:
            object.__setattr__(self, "a_3d", self.a)
        if self.afd_3d is None and self.afd.ndim == 3:
            object.__setattr__(self, "afd_3d", self.afd)

        # Synchronize 4D representations if possible
        if self.a_4d is None and self.a.ndim == 3:
            try:
                nc, ns = self.n_countries, self.n_sectors
                if self.a.shape == (ns * nc, ns, nc):
                    a_4d = self.a.reshape(nc, ns, ns, nc).transpose(1, 0, 2, 3)
                    object.__setattr__(self, "a_4d", a_4d)
            except Exception:
                pass

        if self.afd_4d is None and self.afd.ndim == 3:
            try:
                nc, ns, nfd = self.n_countries, self.n_sectors, self.n_final_demand
                if self.afd.shape == (ns * nc, nfd, nc):
                    afd_4d = self.afd.reshape(nc, ns, nfd, nc).transpose(1, 0, 2, 3)
                    object.__setattr__(self, "afd_4d", afd_4d)
            except Exception:
                pass

    # Backward-compatibility aliases matching MATLAB variable names
    @property
    def KT(self) -> np.ndarray:
        """Alias for k_endow matching MATLAB calibrar.m."""
        return self.k_endow

    @property
    def LT(self) -> np.ndarray:
        """Alias for l_endow matching MATLAB calibrar.m."""
        return self.l_endow

    @property
    def ytot_base(self) -> np.ndarray:
        """Alias for ytot matching MATLAB calibrar.m."""
        return self.ytot

    @property
    def nc(self) -> int:
        """Alias for n_countries matching legacy/compact notation."""
        return self.n_countries

    @property
    def ns(self) -> int:
        """Alias for n_sectors matching legacy/compact notation."""
        return self.n_sectors

    def validate(self) -> dict[str, bool]:
        """Perform comprehensive consistency and accounting checks.

        Returns
        -------
        dict[str, bool]
            Boolean checklist of passed mathematical and accounting checks.
        """
        results: dict[str, bool] = {}

        nc = self.n_countries
        ns = self.n_sectors
        nfd = self.n_final_demand

        # Dimensional checks
        results["dim_a"] = bool(self.a.shape == (ns * nc, ns, nc) or self.a.shape == (ns, nc, ns, nc))
        results["dim_afd"] = bool(self.afd.shape == (ns * nc, nfd, nc) or self.afd.shape == (ns, nc, nfd, nc))
        results["dim_alpha"] = bool(self.alpha.shape == (1, ns, nc))
        results["dim_beta"] = bool(self.beta.shape == (1, ns, nc))
        results["dim_tax"] = bool(self.tax.shape == (1, ns, nc))
        results["dim_ytot"] = bool(self.ytot.shape == (1, ns, nc))
        results["dim_k_endow"] = bool(self.k_endow.shape[-1] == nc)
        results["dim_l_endow"] = bool(self.l_endow.shape[-1] == nc)
        results["dim_invforT"] = bool(self.invforT.shape[-1] == nc)

        # Economic non-negativity
        results["nonneg_ytot"] = bool(np.all(self.ytot >= 0.0))
        results["nonneg_k_endow"] = bool(np.all(self.k_endow >= 0.0))
        results["nonneg_l_endow"] = bool(np.all(self.l_endow >= 0.0))
        results["nonneg_a"] = bool(np.all(self.a >= 0.0))

        # Final demand sourcing coefficients (afd) non-negativity:
        # In official OECD ICIO national accounts, Category 1 (Investment / GFCF)
        # records net changes in inventories, which can legitimately be negative
        # when inventories are drawn down (observed in Lithuania, Ukraine, and
        # Vietnam in data_77c_11s.mat, matching MATLAB calibrar.m). Consumption
        # categories (cat != 1) must remain strictly non-negative, and category
        # sourcing shares must sum to 1.0 for active categories.
        if nfd >= 2:
            consumption_mask = [i for i in range(nfd) if i != 1]
            if self.afd.ndim == 4:
                nonneg_consumption = bool(np.all(self.afd[:, :, consumption_mask, :] >= 0.0))
                afd_sums = np.sum(self.afd, axis=(0, 1))
            else:
                nonneg_consumption = bool(np.all(self.afd[:, consumption_mask, :] >= 0.0))
                afd_sums = np.sum(self.afd, axis=0)
            valid_sums = bool(np.all(np.isclose(afd_sums, 1.0) | np.isclose(afd_sums, 0.0)))
            results["nonneg_afd"] = bool(np.all(self.afd >= 0.0) or (nonneg_consumption and valid_sums))
        else:
            results["nonneg_afd"] = bool(np.all(self.afd >= 0.0))

        # Cobb-Douglas share bounds [0, 1]
        results["alpha_bounds"] = bool(
            np.all(self.alpha >= 0.0) and np.all(self.alpha <= 1.0)
        )

        # Global current accounts sum to zero
        results["global_transfer_balance"] = bool(
            abs(float(np.sum(self.invforT))) < 1e-3
        )

        return results

    def summary(self, detailed: bool = True) -> pd.DataFrame:
        """Produce a publication-ready summary DataFrame of calibrated parameters.

        Parameters
        ----------
        detailed : bool, default True
            If True, returns a per-country table with endowments, factor income,
            net foreign transfers, and gross outputs. If False, returns an
            aggregate world-economy diagnostic table.

        Returns
        -------
        pd.DataFrame
            Formatted summary table.
        """
        nc = self.n_countries
        codes = list(self.country_codes) if len(self.country_codes) == nc else [
            f"C{i:02d}" for i in range(nc)
        ]

        kt_flat = np.asarray(self.k_endow, dtype=float).ravel()
        lt_flat = np.asarray(self.l_endow, dtype=float).ravel()
        inv_flat = np.asarray(self.invforT, dtype=float).ravel()
        ytot_sq = np.asarray(self.ytot, dtype=float).reshape(self.n_sectors, nc)
        beta_sq = np.asarray(self.beta, dtype=float).reshape(self.n_sectors, nc)
        tax_sq = np.asarray(self.tax, dtype=float).reshape(self.n_sectors, nc)

        if detailed:
            data = {
                "Capital_Endow": kt_flat,
                "Labor_Endow": lt_flat,
                "Factor_Income": kt_flat + lt_flat,
                "Net_Transfers": inv_flat,
                "Gross_Output": ytot_sq.sum(axis=0),
                "Mean_TFP": beta_sq.mean(axis=0),
                "Mean_TaxRate": tax_sq.mean(axis=0),
            }
            return pd.DataFrame(data, index=pd.Index(codes, name="Country"))

        records = [
            {"Metric": "Number of Countries", "Value": f"{nc}"},
            {"Metric": "Number of Sectors", "Value": f"{self.n_sectors}"},
            {"Metric": "Final Demand Categories", "Value": f"{self.n_final_demand}"},
            {"Metric": "Global Gross Output", "Value": f"{ytot_sq.sum():.4e}"},
            {"Metric": "Global Capital Endowment", "Value": f"{kt_flat.sum():.4e}"},
            {"Metric": "Global Labor Endowment", "Value": f"{lt_flat.sum():.4e}"},
            {"Metric": "Global Current Account Net Imbalance", "Value": f"{abs(inv_flat.sum()):.4e}"},
            {"Metric": "Mean World TFP Scale", "Value": f"{beta_sq.mean():.4f}"},
            {"Metric": "Mean World Production Tax Rate", "Value": f"{tax_sq.mean():.4f}"},
        ]
        return pd.DataFrame(records).set_index("Metric")

    def to_frame(self, detailed: bool = True) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary(detailed=detailed)

    def to_markdown(self, detailed: bool = True, **kwargs) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(detailed=detailed), **kwargs)

    def to_latex(self, detailed: bool = True, **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(detailed=detailed), **kwargs)

    def to_typst(self, detailed: bool = True, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(detailed=detailed), **kwargs)


@dataclass(frozen=True)
class TradeEquilibriumResult:
    """Equilibrium solution and flow accounting of the multi-country multi-sector CGE model.

    Attributes
    ----------
    x_sol : np.ndarray, shape (2001,)
        Full state vector solution: [log(p); log(y); log(r); log(w); T; XN].
    p_sol : np.ndarray, shape (1, 11, 77)
        Equilibrium gross output prices.
    y_sol : np.ndarray, shape (1, 11, 77)
        Equilibrium gross output quantities.
    r_sol : np.ndarray, shape (1, 1, 77) or (1, 77)
        Equilibrium capital rental rates.
    w_sol : np.ndarray, shape (1, 1, 77) or (1, 77)
        Equilibrium wage rates.
    T_sol : np.ndarray, shape (1, 1, 77) or (1, 77)
        Equilibrium government tax revenues.
    XN_sol : np.ndarray, shape (76,) or (76, 1)
        Equilibrium net foreign transfers for countries 1..76.
    intermediate_flows : np.ndarray | None, shape (11, 77, 11, 77), default None
        4D bilateral intermediate transaction flows tensor.
    final_demand_flows : np.ndarray | None, shape (11, 77, 3, 77), default None
        4D bilateral final demand deliveries tensor.
    p_fd : np.ndarray | None, shape (1, 3, 77), default None
        Composite final demand purchaser prices before consumption taxes.
    c_fd : np.ndarray | None, shape (1, 3, 77), default None
        Composite final demand real absorption volumes.
    gdp : np.ndarray | None, shape (77,), default None
        National GDP at market prices (factor cost plus net taxes and tariffs).
    gdp_fc : np.ndarray | None, shape (77,), default None
        National GDP at factor cost (labor income plus capital income).
    imports : np.ndarray | None, shape (77,), default None
        National total gross imports across intermediate and final goods.
    exports : np.ndarray | None, shape (77,), default None
        National total gross exports across intermediate and final goods.
    tariffs : np.ndarray | None, shape (77,), default None
        National total tariff revenue collections.
    cpi : np.ndarray | None, shape (77,), default None
        Domestic wage-deflated consumer price index relative to baseline.
    terms_of_trade : np.ndarray | None, shape (77,), default None
        National terms of trade index (export price index / import price index).
    c_sol : np.ndarray | None, shape (1, 3, 77), default None
        Alias for c_fd.
    pfd_sol : np.ndarray | None, shape (1, 3, 77), default None
        Alias for p_fd.
    Pfd_final : np.ndarray | None, shape (1, 3, 77), default None
        Final demand tax-inclusive consumer prices.
    qxX0_sol : np.ndarray | None, shape (11, 11, 77, 77), default None
        Bilateral intermediate trade flow quantities in MATLAB axis ordering.
    qxFD0_sol : np.ndarray | None, shape (11, 3, 77, 77), default None
        Bilateral final demand trade flow quantities in MATLAB axis ordering.
    data_model_vf : np.ndarray | None, shape (850, 1078), default None
        Reconstructed equilibrium transaction matrix.
    data_tariff_vf : np.ndarray | None, shape (851, 1078), default None
        Reconstructed equilibrium transaction matrix including tariff collections.
    converged : bool, default True
        Whether the nonlinear solver converged within tolerance.
    iterations : int, default 0
        Number of Newton-Raphson or root iterations completed.
    diff : float, default 0.0
        Final L1 equation residual norm: sum(|ff|).
    residual_norm : float, default 0.0
        Final L1 equation residual norm (alias for diff).
    max_residual : float, default 0.0
        Maximum absolute equation residual: max(|ff|).
    residuals : np.ndarray | None, default None
        Residual vector ff at solution.
    country_codes : tuple[str, ...], default ()
        ISO-3 country identifier codes.
    sector_codes : tuple[str, ...], default ()
        Sector identifier codes.
    metadata : dict[str, Any], default empty
        Solver diagnostics, timing, scenario description.
    """

    x_sol: np.ndarray
    p_sol: np.ndarray
    y_sol: np.ndarray
    r_sol: np.ndarray
    w_sol: np.ndarray
    T_sol: np.ndarray
    XN_sol: np.ndarray
    intermediate_flows: np.ndarray | None = None
    final_demand_flows: np.ndarray | None = None
    p_fd: np.ndarray | None = None
    c_fd: np.ndarray | None = None
    gdp: np.ndarray | None = None
    gdp_fc: np.ndarray | None = None
    imports: np.ndarray | None = None
    exports: np.ndarray | None = None
    tariffs: np.ndarray | None = None
    cpi: np.ndarray | None = None
    terms_of_trade: np.ndarray | None = None
    c_sol: np.ndarray | None = None
    pfd_sol: np.ndarray | None = None
    Pfd_final: np.ndarray | None = None
    qxX0_sol: np.ndarray | None = None
    qxFD0_sol: np.ndarray | None = None
    data_model_vf: np.ndarray | None = None
    data_tariff_vf: np.ndarray | None = None
    converged: bool = True
    iterations: int = 0
    diff: float = 0.0
    residual_norm: float = 0.0
    max_residual: float = 0.0
    residuals: np.ndarray | None = None
    country_codes: tuple[str, ...] = field(default_factory=tuple)
    sector_codes: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required state arrays and synchronize alias attributes."""
        for name, arr in [
            ("x_sol", self.x_sol),
            ("p_sol", self.p_sol),
            ("y_sol", self.y_sol),
            ("r_sol", self.r_sol),
            ("w_sol", self.w_sol),
            ("T_sol", self.T_sol),
            ("XN_sol", self.XN_sol),
        ]:
            if arr is None:
                raise ValueError(f"TradeEquilibriumResult field '{name}' cannot be None.")
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f"TradeEquilibriumResult field '{name}' must be a numpy.ndarray, "
                    f"got {type(arr).__name__}."
                )

        # Synchronize aliases
        if self.c_fd is not None and self.c_sol is None:
            object.__setattr__(self, "c_sol", self.c_fd)
        elif self.c_sol is not None and self.c_fd is None:
            object.__setattr__(self, "c_fd", self.c_sol)

        if self.p_fd is not None and self.pfd_sol is None:
            object.__setattr__(self, "pfd_sol", self.p_fd)
        elif self.pfd_sol is not None and self.p_fd is None:
            object.__setattr__(self, "p_fd", self.pfd_sol)

        if self.residual_norm == 0.0 and self.diff != 0.0:
            object.__setattr__(self, "residual_norm", self.diff)
        elif self.diff == 0.0 and self.residual_norm != 0.0:
            object.__setattr__(self, "diff", self.residual_norm)

        if not isinstance(self.country_codes, tuple):
            object.__setattr__(self, "country_codes", tuple(self.country_codes))
        if not isinstance(self.sector_codes, tuple):
            object.__setattr__(self, "sector_codes", tuple(self.sector_codes))

    def summary(self, detailed: bool = False) -> pd.DataFrame:
        """Produce summary DataFrame of solver convergence or national economic aggregates.

        Parameters
        ----------
        detailed : bool, default False
            If False, returns a high-level solver diagnostic table.
            If True, returns a country-level table of GDP, trade flows, tariffs,
            and terms of trade.

        Returns
        -------
        pd.DataFrame
            Summary table.
        """
        if not detailed:
            norm_val = self.diff if self.diff != 0.0 else self.residual_norm
            records = [
                {"Diagnostic": "Solver Status", "Value": "CONVERGED" if self.converged else "FAILED"},
                {"Diagnostic": "Iterations", "Value": str(self.iterations)},
                {"Diagnostic": "L1 Residual Norm", "Value": f"{norm_val:.6e}"},
                {"Diagnostic": "Max Absolute Residual", "Value": f"{self.max_residual:.6e}"},
                {"Diagnostic": "Mean Price Index", "Value": f"{float(np.mean(self.p_sol)):.6f}"},
                {"Diagnostic": "Total World Output", "Value": f"{float(np.sum(self.y_sol)):.4e}"},
                {"Diagnostic": "Mean Wage Index", "Value": f"{float(np.mean(self.w_sol)):.6f}"},
                {"Diagnostic": "Mean Capital Rental Rate", "Value": f"{float(np.mean(self.r_sol)):.6f}"},
            ]
            return pd.DataFrame(records).set_index("Diagnostic")

        # Detailed country-by-country economic summary
        nc = (
            len(self.gdp)
            if self.gdp is not None
            else (len(self.country_codes) if self.country_codes else (self.p_sol.shape[-1] if self.p_sol.ndim >= 2 else 77))
        )
        codes = list(self.country_codes) if len(self.country_codes) == nc else [f"C{i:02d}" for i in range(nc)]
        data: dict[str, Any] = {}
        if self.gdp is not None:
            data["GDP_Market_Prices"] = np.asarray(self.gdp, dtype=float).ravel()
        if self.gdp_fc is not None:
            data["GDP_Factor_Cost"] = np.asarray(self.gdp_fc, dtype=float).ravel()
        if self.exports is not None:
            data["Exports"] = np.asarray(self.exports, dtype=float).ravel()
        if self.imports is not None:
            data["Imports"] = np.asarray(self.imports, dtype=float).ravel()
        if self.exports is not None and self.imports is not None:
            data["Net_Exports"] = np.asarray(self.exports, dtype=float).ravel() - np.asarray(self.imports, dtype=float).ravel()
        if self.tariffs is not None:
            data["Tariff_Revenue"] = np.asarray(self.tariffs, dtype=float).ravel()
        if self.cpi is not None:
            data["CPI"] = np.asarray(self.cpi, dtype=float).ravel()
        if self.terms_of_trade is not None:
            data["Terms_of_Trade"] = np.asarray(self.terms_of_trade, dtype=float).ravel()

        if not data:
            data["Wage"] = np.asarray(self.w_sol, dtype=float).ravel()
            data["Capital_Rental"] = np.asarray(self.r_sol, dtype=float).ravel()
            data["Transfers"] = np.asarray(self.T_sol, dtype=float).ravel()

        return pd.DataFrame(data, index=pd.Index(codes, name="Country"))

    def to_frame(self, detailed: bool = False) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary(detailed=detailed)

    def to_markdown(self, detailed: bool = False, **kwargs) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(detailed=detailed), **kwargs)

    def to_latex(self, detailed: bool = False, **kwargs) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(detailed=detailed), **kwargs)

    def to_typst(self, detailed: bool = False, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(detailed=detailed), **kwargs)

    @property
    def net_exports(self) -> np.ndarray | None:
        """Net exports (Exports - Imports) by country."""
        if self.exports is not None and self.imports is not None:
            return np.asarray(self.exports, dtype=float) - np.asarray(self.imports, dtype=float)
        return None

    def bilateral_trade_frame(self) -> pd.DataFrame:
        """Return square (nc x nc) bilateral gross trade flows DataFrame."""
        trade_matrix = self.metadata.get("bilateral_trade")
        nc = (
            trade_matrix.shape[0]
            if trade_matrix is not None
            else (len(self.country_codes) if self.country_codes else 77)
        )
        codes = list(self.country_codes) if len(self.country_codes) == nc else [f"C{i:02d}" for i in range(nc)]
        if trade_matrix is None:
            trade_matrix = np.zeros((nc, nc), dtype=float)
        return pd.DataFrame(trade_matrix, index=pd.Index(codes, name="Origin"), columns=codes)

    def sectoral_tariffs_frame(self) -> pd.DataFrame:
        """Return (nc x ns) intermediate tariff revenue collections DataFrame."""
        t_sec = self.metadata.get("tariffs_interm")
        nc = t_sec.shape[0] if t_sec is not None else len(self.country_codes)
        ns = t_sec.shape[1] if t_sec is not None else len(self.sector_codes)
        c_codes = list(self.country_codes) if len(self.country_codes) == nc else [f"C{i:02d}" for i in range(nc)]
        s_codes = list(self.sector_codes) if len(self.sector_codes) == ns else [f"S{j:02d}" for j in range(ns)]
        if t_sec is None:
            t_sec = np.zeros((nc, ns), dtype=float)
        return pd.DataFrame(t_sec, index=pd.Index(c_codes, name="Country"), columns=s_codes)


@dataclass(frozen=True)
class GearyKhamisResult:
    """Multilateral Geary-Khamis Purchasing Power Parity (PPP) solution.

    Attributes
    ----------
    pi : np.ndarray, shape (3,) or (nfd,)
        International reference commodity prices.
    ppp : np.ndarray, shape (77,) or (nc,)
        Purchasing power parity exchange rates per country.
    real_gdp : np.ndarray, shape (77,) or (nc,)
        Real GDP in Geary-Khamis international dollars.
    nominal_gdp : np.ndarray, shape (77,) or (nc,)
        Nominal expenditure in domestic currency / base units.
    world_real_gdp : float
        Sum of real GDP across all countries.
    world_nominal_gdp : float
        Sum of nominal GDP across all countries.
    gdp_growth : np.ndarray | None, shape (77,), default None
        Percentage real GDP growth vs. baseline scenario.
    country_codes : tuple[str, ...], default ()
        ISO-3 country identifier codes.
    category_codes : tuple[str, ...], default ()
        Final demand or sector category identifier codes.
    converged : bool, default True
        Whether the fixed-point iteration or linear solve converged.
    iterations : int, default 0
        Number of fixed-point iterations.
    tolerance : float, default 1e-10
        Convergence tolerance threshold.
    method : str, default "iterative"
        Computational method ('iterative' or 'linear').
    matlab_compat : bool, default True
        Whether MATLAB Resultadosl.m sign convention was used for ROW.
    real_gdp_growth : np.ndarray | None, shape (77,), default None
        Alias for gdp_growth.
    diff : float, default 0.0
        Final L1 difference ||PPP_{t} - PPP_{t-1}||_1.
    metadata : dict[str, Any], default empty
        Additional solver metadata and diagnostics.
    """

    pi: np.ndarray
    ppp: np.ndarray
    real_gdp: np.ndarray
    nominal_gdp: np.ndarray
    world_real_gdp: float = 0.0
    world_nominal_gdp: float = 0.0
    gdp_growth: np.ndarray | None = None
    country_codes: tuple[str, ...] = field(default_factory=tuple)
    category_codes: tuple[str, ...] = field(default_factory=tuple)
    converged: bool = True
    iterations: int = 0
    tolerance: float = 1e-10
    method: str = "iterative"
    matlab_compat: bool = True
    real_gdp_growth: np.ndarray | None = None
    diff: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Synchronize alias attributes and tuple types."""
        if self.gdp_growth is not None and self.real_gdp_growth is None:
            object.__setattr__(self, "real_gdp_growth", self.gdp_growth)
        elif self.real_gdp_growth is not None and self.gdp_growth is None:
            object.__setattr__(self, "gdp_growth", self.real_gdp_growth)

        if self.world_real_gdp == 0.0 and self.real_gdp is not None and len(self.real_gdp) > 0:
            object.__setattr__(self, "world_real_gdp", float(np.sum(self.real_gdp)))
        if self.world_nominal_gdp == 0.0 and self.nominal_gdp is not None and len(self.nominal_gdp) > 0:
            object.__setattr__(self, "world_nominal_gdp", float(np.sum(self.nominal_gdp)))

        if not isinstance(self.country_codes, tuple):
            object.__setattr__(self, "country_codes", tuple(self.country_codes))
        if not isinstance(self.category_codes, tuple):
            object.__setattr__(self, "category_codes", tuple(self.category_codes))

    def summary(self, detailed: bool = True) -> pd.DataFrame:
        """Produce real GDP and PPP summary table."""
        nc = len(self.ppp)
        codes = list(self.country_codes) if len(self.country_codes) == nc else [
            f"C{i:02d}" for i in range(nc)
        ]
        if detailed:
            data: dict[str, Any] = {
                "PPP_Rate": np.asarray(self.ppp, dtype=float).ravel(),
                "Nominal_GDP": np.asarray(self.nominal_gdp, dtype=float).ravel(),
                "Real_GDP_GK": np.asarray(self.real_gdp, dtype=float).ravel(),
            }
            growth = self.gdp_growth if self.gdp_growth is not None else self.real_gdp_growth
            if growth is not None:
                data["Real_GDP_Growth_%"] = np.asarray(growth, dtype=float).ravel()
            return pd.DataFrame(data, index=pd.Index(codes, name="Country"))

        # Aggregate summary
        usa_idx = codes.index("USA") if "USA" in codes else (73 if nc > 73 else -1)
        us_ppp = float(self.ppp[usa_idx]) if usa_idx >= 0 else 1.0
        agg_data = {
            "Metric": [
                "World_Nominal_GDP",
                "World_Real_GDP_GK",
                "Mean_PPP",
                "US_PPP",
                "Converged",
                "Iterations",
                "Tolerance",
                "Method",
            ],
            "Value": [
                f"{self.world_nominal_gdp:,.2f}",
                f"{self.world_real_gdp:,.2f}",
                f"{float(np.mean(self.ppp)):.4f}",
                f"{us_ppp:.4f}",
                str(self.converged),
                str(self.iterations),
                f"{self.tolerance:.1e}",
                self.method,
            ],
        }
        return pd.DataFrame(agg_data).set_index("Metric")

    def to_frame(self, detailed: bool = True) -> pd.DataFrame:
        return self.summary(detailed=detailed)

    def to_markdown(self, detailed: bool = True, **kwargs: Any) -> str:
        return df_to_markdown(self.summary(detailed=detailed), **kwargs)

    def to_latex(self, detailed: bool = True, **kwargs: Any) -> str:
        return df_to_latex(self.summary(detailed=detailed), **kwargs)

    def to_typst(self, detailed: bool = True, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(detailed=detailed), **kwargs)

    def category_prices_frame(self) -> pd.DataFrame:
        """Return international prices labeled by commodity category."""
        nm = len(self.pi)
        cat_labels = list(self.category_codes) if len(self.category_codes) == nm else [
            f"Category_{i+1}" for i in range(nm)
        ]
        return pd.DataFrame(
            {"International_Price_Pi": np.asarray(self.pi, dtype=float).ravel()},
            index=pd.Index(cat_labels, name="Category"),
        )


def _restore_xn(
    c: np.ndarray,
    xn: np.ndarray,
    nc: int,
    matlab_compat: bool = True,
) -> np.ndarray:
    """Helper to restore foreign investment into Category 2 of final demand."""
    c_arr = np.asarray(c, dtype=float).copy()
    xn_flat = np.asarray(xn, dtype=float).ravel()
    xn_full = np.zeros(nc, dtype=float)
    if len(xn_flat) == nc - 1:
        xn_full[: nc - 1] = xn_flat
        xn_full[nc - 1] = float(np.sum(xn_flat)) if matlab_compat else -float(np.sum(xn_flat))
    elif len(xn_flat) == nc:
        xn_full = xn_flat.copy()
        if matlab_compat and xn_full[nc - 1] < 0 and np.sum(xn_full[: nc - 1]) > 0:
            xn_full[nc - 1] = float(np.sum(xn_full[: nc - 1]))
        elif not matlab_compat and xn_full[nc - 1] > 0 and np.sum(xn_full[: nc - 1]) > 0:
            xn_full[nc - 1] = -float(np.sum(xn_full[: nc - 1]))
    if c_arr.ndim == 3:
        c_arr[..., 1, :] += xn_full
    elif c_arr.ndim == 2:
        c_arr[1, :] += xn_full
    return c_arr


@dataclass(frozen=True)
class ScenarioBatchResult:
    """Container for comparative multi-scenario counterfactual evaluations.

    Attributes
    ----------
    scenarios : dict[str, TradeEquilibriumResult]
        Equilibrium results keyed by scenario name (e.g. 'base', 't10', 't10_25').
    geary_khamis : dict[str, GearyKhamisResult]
        Geary-Khamis multilateral PPP results keyed by scenario name.
    baseline_scenario : str, default 'base'
        Identifier of the baseline benchmark scenario.
    country_codes : tuple[str, ...], default ()
        Tuple of ISO-3 country codes.
    sector_codes : tuple[str, ...], default ()
        Tuple of sector codes.
    metadata : dict[str, Any], default empty
        Orchestration metadata and solver diagnostics.
    ppp_results : dict[str, GearyKhamisResult]
        Alias for geary_khamis.
    real_gdp_table : pd.DataFrame
        Percentage real GDP growth table across scenarios and countries.
    trade_balance_table : pd.DataFrame
        Trade balance / net export changes across scenarios and countries.
    cpi_table : pd.DataFrame
        Domestic wage-deflated consumer price index inflation across scenarios.
    """

    scenarios: dict[str, TradeEquilibriumResult]
    geary_khamis: dict[str, GearyKhamisResult] = field(default_factory=dict)
    baseline_scenario: str = "base"
    country_codes: tuple[str, ...] = field(default_factory=tuple)
    sector_codes: tuple[str, ...] = field(default_factory=tuple)
    metadata: dict[str, Any] = field(default_factory=dict)
    ppp_results: dict[str, GearyKhamisResult] = field(default_factory=dict)
    real_gdp_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_balance_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    cpi_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self) -> None:
        """Synchronize ppp_results/geary_khamis and comparative tables."""
        if self.geary_khamis and not self.ppp_results:
            object.__setattr__(self, "ppp_results", self.geary_khamis)
        elif self.ppp_results and not self.geary_khamis:
            object.__setattr__(self, "geary_khamis", self.ppp_results)

        if not self.country_codes and self.scenarios:
            first_res = next(iter(self.scenarios.values()))
            if first_res.country_codes:
                object.__setattr__(self, "country_codes", tuple(first_res.country_codes))
            if first_res.sector_codes and not self.sector_codes:
                object.__setattr__(self, "sector_codes", tuple(first_res.sector_codes))

        if self.real_gdp_table.empty and (self.geary_khamis or self.scenarios):
            cols = list(self.scenarios.keys())
            nc = len(self.country_codes) if self.country_codes else 77
            c_idx = (
                pd.Index(list(self.country_codes), name="Country")
                if self.country_codes
                else pd.Index([f"C{i}" for i in range(nc)], name="Country")
            )

            gdp_mat = np.zeros((nc, len(cols)), dtype=float)
            cpi_mat = np.zeros((nc, len(cols)), dtype=float)
            tb_mat = np.zeros((nc, len(cols)), dtype=float)

            base_res = self.scenarios.get(self.baseline_scenario)
            base_gdp = None
            matlab_compat = bool(self.metadata.get("matlab_compat", True))
            if base_res is not None and base_res.c_sol is not None:
                c_b = _restore_xn(base_res.c_sol, base_res.XN_sol, nc, matlab_compat=matlab_compat)
                base_gdp = np.sum(c_b[0] if c_b.ndim == 3 else c_b, axis=0)

            for j, s_name in enumerate(cols):
                res = self.scenarios[s_name]
                if s_name in self.geary_khamis:
                    gk = self.geary_khamis[s_name]
                    growth = gk.gdp_growth if gk.gdp_growth is not None else gk.real_gdp_growth
                    if growth is not None:
                        gdp_mat[:, j] = np.asarray(growth, dtype=float).ravel()

                if res.cpi is not None:
                    cpi_mat[:, j] = np.asarray(res.cpi, dtype=float).ravel()
                elif base_gdp is not None and res.pfd_sol is not None and res.w_sol is not None and res.c_sol is not None:
                    c_s = _restore_xn(res.c_sol, res.XN_sol, nc, matlab_compat=matlab_compat)
                    p_s = res.pfd_sol[0] if res.pfd_sol.ndim == 3 else res.pfd_sol
                    w_s = res.w_sol.ravel()
                    p_dom = p_s / w_s[np.newaxis, :]
                    gdp_dom = np.sum(p_dom * (c_s[0] if c_s.ndim == 3 else c_s), axis=0)
                    cpi_mat[:, j] = gdp_dom / base_gdp
                else:
                    cpi_mat[:, j] = 1.0

                if res.exports is not None and res.imports is not None:
                    tb_mat[:, j] = np.asarray(res.exports - res.imports, dtype=float).ravel()
                elif res.XN_sol is not None:
                    xn = np.asarray(res.XN_sol, dtype=float).ravel()
                    xn_full = np.zeros(nc)
                    xn_full[:min(len(xn), nc)] = xn[:min(len(xn), nc)]
                    if len(xn) == nc - 1:
                        xn_full[nc - 1] = np.sum(xn) if matlab_compat else -np.sum(xn)
                    tb_mat[:, j] = xn_full

            object.__setattr__(self, "real_gdp_table", pd.DataFrame(gdp_mat, index=c_idx, columns=cols))
            object.__setattr__(self, "cpi_table", pd.DataFrame(cpi_mat, index=c_idx, columns=cols))
            object.__setattr__(self, "trade_balance_table", pd.DataFrame(tb_mat, index=c_idx, columns=cols))

    def __getitem__(self, key: str) -> TradeEquilibriumResult:
        return self.scenarios[key]

    def __contains__(self, key: str) -> bool:
        return key in self.scenarios

    def __iter__(self):
        return iter(self.scenarios)

    def __len__(self) -> int:
        return len(self.scenarios)

    def keys(self):
        return self.scenarios.keys()

    def values(self):
        return self.scenarios.values()

    def items(self):
        return self.scenarios.items()

    def with_geary_khamis(
        self, geary_khamis: dict[str, GearyKhamisResult]
    ) -> ScenarioBatchResult:
        """Return a new ScenarioBatchResult with updated Geary-Khamis PPP results."""
        from dataclasses import replace

        return replace(
            self,
            geary_khamis=geary_khamis,
            ppp_results=geary_khamis,
            real_gdp_table=pd.DataFrame(),
            cpi_table=pd.DataFrame(),
            trade_balance_table=pd.DataFrame(),
        )

    def to_frame(self, metric: str = "real_gdp_growth") -> pd.DataFrame:
        """Return comparative DataFrame of countries across scenarios."""
        metric_norm = metric.lower().strip()
        if metric_norm in ("real_gdp_growth", "gdp_growth", "gdp", "growth"):
            return self.real_gdp_table
        if metric_norm in ("cpi", "price_index"):
            return self.cpi_table
        if metric_norm in ("inflation", "cpi_inflation"):
            return (self.cpi_table - 1.0) * 100.0
        if metric_norm in ("trade_balance", "net_exports", "xn"):
            return self.trade_balance_table
        if metric_norm in ("xn_over_gdp", "xn_gdp"):
            data: dict[str, np.ndarray] = {}
            nc = len(self.country_codes) if self.country_codes else len(self.real_gdp_table)
            base_res = self.scenarios.get(self.baseline_scenario)
            base_gdp = None
            matlab_compat = bool(self.metadata.get("matlab_compat", True))
            if base_res is not None and base_res.c_sol is not None:
                c_b = _restore_xn(base_res.c_sol, base_res.XN_sol, nc, matlab_compat=matlab_compat)
                base_gdp = np.sum(c_b[0] if c_b.ndim == 3 else c_b, axis=0)

            for s_name, res in self.scenarios.items():
                if res.XN_sol is not None:
                    xn = np.asarray(res.XN_sol, dtype=float).ravel()
                    xn_full = np.zeros(nc)
                    xn_full[:min(len(xn), nc)] = xn[:min(len(xn), nc)]
                    if len(xn) == nc - 1:
                        xn_full[nc - 1] = np.sum(xn) if matlab_compat else -np.sum(xn)

                    if s_name == self.baseline_scenario and base_gdp is not None:
                        gdp_val = base_gdp
                    elif res.pfd_sol is not None and res.w_sol is not None and res.c_sol is not None:
                        c_s = _restore_xn(res.c_sol, res.XN_sol, nc, matlab_compat=matlab_compat)
                        p_s = res.pfd_sol[0] if res.pfd_sol.ndim == 3 else res.pfd_sol
                        w_s = res.w_sol.ravel()
                        p_dom = p_s / w_s[np.newaxis, :]
                        gdp_val = np.sum(p_dom * (c_s[0] if c_s.ndim == 3 else c_s), axis=0)
                    elif res.gdp is not None:
                        gdp_val = res.gdp
                    elif base_gdp is not None:
                        gdp_val = base_gdp
                    else:
                        gdp_val = np.ones(nc)

                    data[s_name] = 100.0 * xn_full / gdp_val
            return pd.DataFrame(data, index=self.real_gdp_table.index)

        raise ValueError(
            f"Unknown metric '{metric}'. Valid options: 'real_gdp_growth', 'cpi', "
            f"'inflation', 'trade_balance', 'xn_over_gdp'."
        )

    def to_selected_country_table(
        self,
        countries: Sequence[str] | None = None,
        round_digits: int = 2,
    ) -> pd.DataFrame:
        """Reproduce selected economies impact table (selected_country_impacts.tex)."""
        from puremacro.trade.data import EU_COUNTRY_CODES

        if countries is None:
            c_list = ["CAN", "CHN", "EUR", "MEX", "USA"]
        else:
            c_list = list(countries)

        all_countries = list(self.country_codes) if self.country_codes else list(self.real_gdp_table.index)
        if not self.scenarios or not all_countries:
            idx = pd.MultiIndex.from_tuples([], names=["Section", "Country"])
            return pd.DataFrame(index=idx)

        is_eu = np.array([c in EU_COUNTRY_CODES for c in all_countries], dtype=bool)

        base_res = self.scenarios.get(self.baseline_scenario)
        matlab_compat = bool(self.metadata.get("matlab_compat", True))
        if base_res is not None and base_res.c_sol is not None:
            c_base = _restore_xn(base_res.c_sol, base_res.XN_sol, len(all_countries), matlab_compat=matlab_compat)
            gdp_b = np.sum(c_base[0] if c_base.ndim == 3 else c_base, axis=0).ravel()
        else:
            gdp_b = np.ones(len(all_countries))

        sum_eu_gdp = np.sum(gdp_b[is_eu])
        w_eu = gdp_b[is_eu] / sum_eu_gdp if sum_eu_gdp > 0 else np.ones(np.sum(is_eu)) / max(np.sum(is_eu), 1)

        cf_scens = [s for s in self.scenarios.keys() if s != self.baseline_scenario]
        if not cf_scens:
            cf_scens = list(self.scenarios.keys())

        def _get_vals(metric_df: pd.DataFrame, c_code: str) -> list[float]:
            if c_code in ("EUR", "EU_"):
                if np.any(is_eu) and not metric_df.empty:
                    eu_sub = metric_df.iloc[is_eu]
                    return [float(np.sum(w_eu * eu_sub[s])) for s in cf_scens]
                return [0.0] * len(cf_scens)
            if c_code in metric_df.index:
                return [float(metric_df.loc[c_code, s]) for s in cf_scens]
            return [0.0] * len(cf_scens)

        rows: list[dict[str, Any]] = []

        # Section 1: GDP growth
        for c in c_list:
            vals = _get_vals(self.real_gdp_table, c)
            row: dict[str, Any] = {"Section": "GDP growth (%)", "Country": c}
            for s, v in zip(cf_scens, vals):
                row[s] = round(v, round_digits)
            row["Base"] = np.nan
            rows.append(row)

        # Section 2: Inflation
        infl_table = (self.cpi_table - 1.0) * 100.0 if not self.cpi_table.empty else pd.DataFrame()
        for c in c_list:
            vals = _get_vals(infl_table, c)
            row_inf: dict[str, Any] = {"Section": "Inflation (%)", "Country": c}
            for s, v in zip(cf_scens, vals):
                row_inf[s] = round(v, round_digits)
            row_inf["Base"] = np.nan
            rows.append(row_inf)

        # Section 3: Net exports / GDP
        xn_gdp_df = self.to_frame("xn_over_gdp")
        for c in c_list:
            vals = _get_vals(xn_gdp_df, c)
            row_xn: dict[str, Any] = {"Section": "Net exports / GDP (%)", "Country": c}
            for s, v in zip(cf_scens, vals):
                row_xn[s] = round(v, round_digits)
            if self.baseline_scenario in xn_gdp_df.columns:
                if c in ("EUR", "EU_") and np.any(is_eu):
                    base_xn_eu = xn_gdp_df.iloc[is_eu][self.baseline_scenario]
                    row_xn["Base"] = round(float(np.sum(w_eu * base_xn_eu)), round_digits)
                elif c in xn_gdp_df.index:
                    row_xn["Base"] = round(float(xn_gdp_df.loc[c, self.baseline_scenario]), round_digits)
                else:
                    row_xn["Base"] = np.nan
            else:
                row_xn["Base"] = np.nan
            rows.append(row_xn)

        return pd.DataFrame(rows).set_index(["Section", "Country"])

    def to_mean_by_scenario_table(self, round_digits: int = 3) -> pd.DataFrame:
        """Reproduce cross-country mean and standard deviation table (mean_by_scenario.tex)."""
        cf_scens = [s for s in self.scenarios.keys() if s != self.baseline_scenario]
        if not cf_scens:
            cf_scens = list(self.scenarios.keys())

        gdp_df = self.real_gdp_table[cf_scens]
        infl_df = ((self.cpi_table - 1.0) * 100.0)[cf_scens]
        xn_df = self.to_frame("xn_over_gdp")[cf_scens]

        records: list[dict[str, Any]] = []
        for label, metric_df in [
            ("GDP growth (%)", gdp_df),
            ("Inflation (%)", infl_df),
            ("XN over GDP (%)", xn_df),
        ]:
            row = {"Variable": label}
            for s in cf_scens:
                m = float(np.mean(metric_df[s]))
                sd = float(np.std(metric_df[s], ddof=1))
                row[s] = f"{m:.{round_digits}f} ({sd:.{round_digits}f})"
            records.append(row)

        return pd.DataFrame(records).set_index("Variable")

    def summary(self) -> pd.DataFrame:
        """Return summary table of selected economies across scenarios."""
        return self.to_selected_country_table()

    def to_markdown(self, table_type: str = "selected", **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        if table_type in ("mean", "mean_by_scenario"):
            return df_to_markdown(self.to_mean_by_scenario_table(), **kwargs)
        return df_to_markdown(self.to_selected_country_table(), **kwargs)

    def to_latex(self, table_type: str = "selected", **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        if table_type in ("mean", "mean_by_scenario"):
            return df_to_latex(self.to_mean_by_scenario_table(), **kwargs)
        return df_to_latex(self.to_selected_country_table(), **kwargs)

    def to_typst(self, table_type: str = "selected", **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        if table_type in ("mean", "mean_by_scenario"):
            return df_to_typst(self.to_mean_by_scenario_table(), **kwargs)
        return df_to_typst(self.to_selected_country_table(), **kwargs)


@dataclass(frozen=True)
class RetaliationGameResult:
    """Equilibrium outcome of an endogenous foreign retaliation game (Extension A / F20).

    Attributes
    ----------
    scenario_name : str
        Identifier of the simulated scenario.
    equilibrium : TradeEquilibriumResult
        Final CGE general equilibrium solution after retaliation convergence.
    initial_scenario : Any
        Pre-retaliation trigger tariff scenario.
    tau_final : np.ndarray
        Final 4D intermediate tariff multipliers: (ns, nc, ns, nc).
    tau_fd_final : np.ndarray
        Final 4D final demand tariff multipliers: (ns, nc, nfd, nc).
    strategic_players : tuple[str, ...]
        Tuple of active strategic partner country codes.
    partner_tariffs : dict[str, dict[str, float]]
        Effective retaliatory tariff rates imposed on US varieties by partner and sector.
    partner_duties_collected : dict[str, float]
        Total tariff revenue collected by each strategic partner on US exports.
    us_duties_collected : dict[str, float]
        Total tariff revenue collected by the US on imports from each partner.
    outer_iterations : int
        Number of Gauss-Seidel outer loop iterations to reach fixed point.
    converged : bool
        Whether the outer policy game converged within outer_tol.
    outer_error : float
        Final infinity-norm difference between successive tariff tensors.
    mode : str, default "targeted"
        Retaliation policy schedule ('targeted', 'symmetric', 'wto_rebalance').
    metadata : dict[str, Any], default empty
        Optional provenance and diagnostic metadata.
    """

    scenario_name: str
    equilibrium: TradeEquilibriumResult
    initial_scenario: Any
    tau_final: np.ndarray
    tau_fd_final: np.ndarray
    strategic_players: tuple[str, ...]
    partner_tariffs: dict[str, dict[str, float]]
    partner_duties_collected: dict[str, float]
    us_duties_collected: dict[str, float]
    outer_iterations: int
    converged: bool
    outer_error: float
    mode: str = "targeted"
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of partner retaliation rates and duties."""
        records: list[dict[str, Any]] = []
        for p in self.strategic_players:
            tariffs = self.partner_tariffs.get(p, {})
            mean_tariff = float(np.mean(list(tariffs.values()))) if tariffs else 0.0
            max_tariff = float(np.max(list(tariffs.values()))) if tariffs else 0.0
            p_duties = float(self.partner_duties_collected.get(p, 0.0))
            us_duties = float(self.us_duties_collected.get(p, 0.0))
            records.append({
                "Partner": p,
                "Mean_Retaliatory_Tariff": mean_tariff,
                "Max_Retaliatory_Tariff": max_tariff,
                "Partner_Duties_Collected": p_duties,
                "US_Duties_Collected": us_duties,
                "Duty_Parity_Ratio": (p_duties / us_duties) if us_duties > 1e-12 else 0.0,
            })
        return pd.DataFrame(records).set_index("Partner")

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    @property
    def final_tariffs(self) -> np.ndarray:
        """Alias for tau_final."""
        return self.tau_final


@dataclass(frozen=True)
class JCurveDynamicResult:
    """Sequential dynamic quarterly trajectory of trade balance and welfare (Extension B / F21).

    Attributes
    ----------
    scenario_name : str
        Identifier of the simulated scenario.
    quarters : tuple[float, ...]
        Quarter indices (e.g. 0.0, 1.0, 2.0, ..., T).
    sigma_path : tuple[float, ...]
        Intermediate elasticity of substitution path sigma(t).
    t_star : float
        Exact closed-form J-curve turning point horizon t* = -ln(1 - 1/sigma_long)/gamma.
    half_life : float
        Contractual rigidity half-life t_{1/2} = ln(2)/gamma.
    trade_balance_path : np.ndarray
        US nominal net export trajectory across quarters.
    us_exports_path : np.ndarray
        US gross export trajectory across quarters.
    us_imports_path : np.ndarray
        US gross import trajectory across quarters.
    us_gdp_path : np.ndarray
        US real GDP index trajectory across quarters.
    us_cpi_path : np.ndarray
        US consumer price index trajectory across quarters.
    equilibria : tuple[TradeEquilibriumResult, ...]
        Sequence of solved CGE equilibrium results for each quarter.
    gamma : float, default 0.15
        Speed of supply-chain recontracting parameter.
    sigma_long : float, default 4.0
        Long-run Armington elasticity of substitution.
    analytical_trade_balance_path : np.ndarray | None, default None
        Closed-form analytical trade balance trajectory from Theorem 2.
    analytical_imports_path : np.ndarray | None, default None
        Closed-form analytical import expenditure path from Theorem 2.
    metadata : dict[str, Any], default empty
        Optional provenance and diagnostic metadata.
    """

    scenario_name: str
    quarters: tuple[float, ...]
    sigma_path: tuple[float, ...]
    t_star: float
    half_life: float
    trade_balance_path: np.ndarray
    us_exports_path: np.ndarray
    us_imports_path: np.ndarray
    us_gdp_path: np.ndarray
    us_cpi_path: np.ndarray
    equilibria: tuple[TradeEquilibriumResult, ...]
    gamma: float = 0.15
    sigma_long: float = 4.0
    analytical_trade_balance_path: np.ndarray | None = None
    analytical_imports_path: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of quarterly macroeconomic transition."""
        data = {
            "Quarter": list(self.quarters),
            "Sigma": list(self.sigma_path),
            "US_Net_Exports": np.asarray(self.trade_balance_path, dtype=float),
            "US_Exports": np.asarray(self.us_exports_path, dtype=float),
            "US_Imports": np.asarray(self.us_imports_path, dtype=float),
            "US_Real_GDP": np.asarray(self.us_gdp_path, dtype=float),
            "US_CPI": np.asarray(self.us_cpi_path, dtype=float),
        }
        return pd.DataFrame(data).set_index("Quarter")

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    @property
    def exports_path(self) -> np.ndarray:
        """Alias for us_exports_path."""
        return self.us_exports_path

    @property
    def imports_path(self) -> np.ndarray:
        """Alias for us_imports_path."""
        return self.us_imports_path

    @property
    def welfare_path(self) -> np.ndarray:
        """Alias for us_gdp_path."""
        return self.us_gdp_path


@dataclass(frozen=True)
class RevenueRecyclingResult:
    """Equilibrium impacts and welfare decomposition under alternative fiscal closures (Extension C / F22).

    Attributes
    ----------
    scenario_name : str
        Identifier of the simulated scenario.
    closure : str
        Fiscal regime: 'lump_sum', 'capital_tax', 'labor_tax', 'strategic_subsidy', 'deficit_reduction'.
    equilibrium : TradeEquilibriumResult
        Solved CGE equilibrium result under the fiscal closure.
    tariff_revenue : float
        Total tariff revenue collected by the target country.
    household_transfer : float
        Amount returned directly to domestic households.
    factor_tax_cut : float
        Percentage factor tax rate cut (Delta t_K or Delta t_L).
    subsidy_rate : float
        Targeted manufacturing output subsidy rate.
    welfare_decomposition : dict[str, float]
        Three-way Harberger welfare decomposition:
        - 'terms_of_trade': TOT effect
        - 'deadweight_loss': deadweight distortion loss
        - 'fiscal_dividend': double dividend / efficiency gain
        - 'net_welfare_change': net equivalent welfare change
    target_country : str, default "USA"
        Target country applying the fiscal recycling regime.
    metadata : dict[str, Any], default empty
        Optional provenance and diagnostic metadata.
    """

    scenario_name: str
    closure: str
    equilibrium: TradeEquilibriumResult
    tariff_revenue: float
    household_transfer: float
    factor_tax_cut: float
    subsidy_rate: float
    welfare_decomposition: dict[str, float]
    target_country: str = "USA"
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of fiscal allocation and welfare decomposition."""
        records = [
            {"Metric": "Fiscal Closure", "Value": self.closure},
            {"Metric": "Target Country", "Value": self.target_country},
            {"Metric": "Tariff Revenue Collected", "Value": f"{self.tariff_revenue:.4f}"},
            {"Metric": "Household Transfer", "Value": f"{self.household_transfer:.4f}"},
            {"Metric": "Factor Tax Relief", "Value": f"{self.factor_tax_cut * 100.0:.2f}%"},
            {"Metric": "Manufacturing Subsidy Rate", "Value": f"{self.subsidy_rate * 100.0:.2f}%"},
            {"Metric": "Terms of Trade Impact", "Value": f"{self.welfare_decomposition.get('terms_of_trade', 0.0):.4f}"},
            {"Metric": "Deadweight Loss (DWL)", "Value": f"{self.welfare_decomposition.get('deadweight_loss', 0.0):.4f}"},
            {"Metric": "Fiscal Efficiency Dividend", "Value": f"{self.welfare_decomposition.get('fiscal_dividend', 0.0):.4f}"},
            {"Metric": "Net Welfare Change", "Value": f"{self.welfare_decomposition.get('net_welfare_change', 0.0):.4f}"},
        ]
        return pd.DataFrame(records).set_index("Metric")

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    @property
    def welfare_change(self) -> float:
        """Net equivalent welfare change."""
        return float(self.welfare_decomposition.get("net_welfare_change", 0.0))

    @property
    def harberger_decomposition(self) -> dict[str, float]:
        """Alias for welfare_decomposition."""
        return dict(self.welfare_decomposition)


@dataclass(frozen=True)
class CapacityBottleneckResult:
    """Equilibrium impacts under upstream production capacity bottlenecks (Extension D / F23).

    Attributes
    ----------
    scenario_name : str
        Identifier of the simulated scenario.
    equilibrium : TradeEquilibriumResult
        Solved CGE equilibrium under capacity constraints.
    capacity_margins : dict[str, float]
        Spare capacity margins (kappa) by constrained sector.
    capacity_limits : dict[str, float]
        Gross output capacity ceilings y_bar = (1 + kappa) * y0.
    output_levels : dict[str, float]
        Equilibrium gross output levels y_sol by sector.
    capacity_utilization : dict[str, float]
        Capacity utilization ratios (y_sol / y_bar).
    price_escalation : dict[str, float]
        Domestic producer price inflation (p_sol - 1.0) * 100.
    penalty_multipliers : dict[str, float]
        Barrier penalty multipliers Phi(y) / c_V - 1.0.
    target_country : str, default "USA"
        Country subject to domestic capacity constraints.
    penalty_scale : float, default 0.05
        Zeta parameter in smooth penalty function.
    penalty_exponent : float, default 8.0
        Eta parameter in smooth penalty function.
    metadata : dict[str, Any], default empty
        Optional provenance and diagnostic metadata.
    """

    scenario_name: str
    equilibrium: TradeEquilibriumResult
    capacity_margins: dict[str, float]
    capacity_limits: dict[str, float]
    output_levels: dict[str, float]
    capacity_utilization: dict[str, float]
    price_escalation: dict[str, float]
    penalty_multipliers: dict[str, float]
    target_country: str = "USA"
    penalty_scale: float = 0.05
    penalty_exponent: float = 8.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of sectoral capacity and price escalation."""
        records: list[dict[str, Any]] = []
        for sec in sorted(self.capacity_margins.keys()):
            records.append({
                "Sector": sec,
                "Spare_Capacity_Margin": f"{self.capacity_margins.get(sec, 0.0) * 100.0:.1f}%",
                "Capacity_Limit": self.capacity_limits.get(sec, 0.0),
                "Gross_Output": self.output_levels.get(sec, 0.0),
                "Utilization_Rate": f"{self.capacity_utilization.get(sec, 0.0) * 100.0:.2f}%",
                "Producer_Price_Inflation": f"{self.price_escalation.get(sec, 0.0):.2f}%",
                "Barrier_Cost_Penalty": f"{self.penalty_multipliers.get(sec, 0.0) * 100.0:.3f}%",
            })
        return pd.DataFrame(records).set_index("Sector")

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)

    @property
    def capacity_ceilings(self) -> dict[str, float]:
        """Alias for capacity_limits."""
        return dict(self.capacity_limits)

    @property
    def max_penalty_cost(self) -> float:
        """Maximum penalty multiplier across all constrained sectors."""
        return float(max(self.penalty_multipliers.values())) if self.penalty_multipliers else 0.0



@dataclass(frozen=True)
class OptimalTariffResult:
    """Result container for unilateral terms-of-trade optimal tariff optimization.

    Attributes
    ----------
    country_code : str
        Country setting the optimal tariff (e.g. 'USA').
    optimal_tariff_rate : float
        Optimal ad-valorem tariff rate (e.g. 0.184 for 18.4%).
    welfare_gain_pct : float
        Percentage welfare gain relative to baseline: (W^* - W_0) / W_0 * 100.
    baseline_welfare : float
        Baseline welfare value.
    optimal_welfare : float
        Equilibrium welfare under optimal tariff.
    welfare_metric : str
        Welfare objective metric ('geary_khamis', 'equivalent_variation', 'terms_of_trade').
    terms_of_trade_initial : float
        Baseline terms of trade index P_X / P_M.
    terms_of_trade_optimal : float
        Optimal terms of trade index P_X / P_M.
    tariff_grid : np.ndarray
        Array of tariff rates evaluated during grid search / optimization.
    welfare_curve : np.ndarray
        Array of corresponding welfare values for each tariff rate on the grid.
    equilibrium: TradeEquilibriumResult
        Solved CGE equilibrium at the optimal tariff rate.
    target_countries : tuple[str, ...] | None, default None
        Target foreign countries subject to the optimal tariff. If None, universal.
    metadata : dict[str, Any], default empty
        Additional diagnostics, optimization status, and timing.
    """

    country_code: str
    optimal_tariff_rate: float
    welfare_gain_pct: float
    baseline_welfare: float
    optimal_welfare: float
    welfare_metric: str
    terms_of_trade_initial: float
    terms_of_trade_optimal: float
    tariff_grid: np.ndarray
    welfare_curve: np.ndarray
    equilibrium: TradeEquilibriumResult
    target_countries: tuple[str, ...] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def optimal_tariff_pct(self) -> float:
        """Optimal tariff rate as percentage (optimal_tariff_rate * 100.0)."""
        return self.optimal_tariff_rate * 100.0

    @property
    def terms_of_trade_change_pct(self) -> float:
        """Percentage change in terms of trade."""
        if self.terms_of_trade_initial != 0.0:
            return ((self.terms_of_trade_optimal - self.terms_of_trade_initial) / self.terms_of_trade_initial) * 100.0
        return 0.0

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of optimal tariff and macroeconomic impacts."""
        records = [
            {"Metric": "Country", "Value": self.country_code},
            {"Metric": "Optimal Tariff Rate", "Value": f"{self.optimal_tariff_pct:.2f}%"},
            {"Metric": "Welfare Metric", "Value": self.welfare_metric},
            {"Metric": "Baseline Welfare", "Value": f"{self.baseline_welfare:.4f}"},
            {"Metric": "Optimal Welfare", "Value": f"{self.optimal_welfare:.4f}"},
            {"Metric": "Welfare Gain", "Value": f"{self.welfare_gain_pct:+.4f}%"},
            {"Metric": "Initial Terms of Trade", "Value": f"{self.terms_of_trade_initial:.4f}"},
            {"Metric": "Optimal Terms of Trade", "Value": f"{self.terms_of_trade_optimal:.4f}"},
            {"Metric": "Terms of Trade Shift", "Value": f"{self.terms_of_trade_change_pct:+.4f}%"},
            {
                "Metric": "Target Partners",
                "Value": "Universal" if self.target_countries is None else ", ".join(self.target_countries),
            },
        ]
        return pd.DataFrame(records).set_index("Metric")

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)


@dataclass(frozen=True)
class NashTariffResult:
    """Result container for multilateral non-cooperative Nash tariff equilibrium.

    Attributes
    ----------
    strategic_players : tuple[str, ...]
        Tuple of strategic player country/bloc codes (e.g. ('USA', 'CHN', 'EUR', 'CAN', 'MEX')).
    nash_tariffs : dict[str, float]
        Equilibrium ad-valorem tariff rates chosen by each player in the Nash equilibrium.
    welfare_changes_pct : dict[str, float]
        Percentage welfare change for each strategic player relative to Free Trade.
    terms_of_trade_changes_pct : dict[str, float]
        Percentage terms of trade change for each player relative to Free Trade.
    world_welfare_change_pct : float
        Percentage change in aggregate global real welfare.
    outer_iterations : int
        Number of outer policy iterations executed.
    converged : bool
        Whether policy iteration converged within tolerance.
    outer_error : float
        Final infinity norm policy adjustment ||tau^(k) - tau^(k-1)||_inf.
    equilibrium : TradeEquilibriumResult
        Solved CGE general equilibrium at the Nash tariff vector.
    tau_nash : np.ndarray
        Final 4D intermediate tariff multipliers: (ns, nc, ns, nc).
    tau_fd_nash : np.ndarray
        Final 4D final demand tariff multipliers: (ns, nc, nfd, nc).
    policy_mode : str, default 'universal'
        Strategic tariff mode ('universal', 'bilateral', 'sectoral').
    welfare_metric : str, default 'geary_khamis'
        Sovereign welfare objective function metric.
    method : str, default 'best_response'
        Solution algorithm ('best_response' or 'gradient').
    player_welfares : dict[str, float], default empty
        Equilibrium welfare levels at the Nash equilibrium.
    baseline_welfares : dict[str, float], default empty
        Baseline welfare levels under cooperative Free Trade (0% tariffs).
    iteration_history : list[dict[str, Any]], default empty
        Convergence trace per iteration.
    metadata : dict[str, Any], default empty
        Solver diagnostics, parameters, and timing.
    """

    strategic_players: tuple[str, ...]
    nash_tariffs: dict[str, float]
    welfare_changes_pct: dict[str, float]
    terms_of_trade_changes_pct: dict[str, float]
    world_welfare_change_pct: float
    outer_iterations: int
    converged: bool
    outer_error: float
    equilibrium: TradeEquilibriumResult
    tau_nash: np.ndarray
    tau_fd_nash: np.ndarray
    policy_mode: str = "universal"
    welfare_metric: str = "geary_khamis"
    method: str = "best_response"
    player_welfares: dict[str, float] = field(default_factory=dict)
    baseline_welfares: dict[str, float] = field(default_factory=dict)
    iteration_history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Produce publication-ready summary DataFrame of Nash tariffs and player welfare."""
        records: list[dict[str, Any]] = []
        for p in self.strategic_players:
            t_val = self.nash_tariffs.get(p, 0.0)
            t_pct = t_val * 100.0 if isinstance(t_val, (int, float)) else np.nan
            w_chg = self.welfare_changes_pct.get(p, 0.0)
            tot_chg = self.terms_of_trade_changes_pct.get(p, 0.0)
            records.append({
                "Player": p,
                "Nash_Tariff_%": f"{t_pct:.2f}%",
                "Welfare_Change_%": f"{w_chg:+.2f}%",
                "Terms_of_Trade_Shift_%": f"{tot_chg:+.2f}%",
            })
        return pd.DataFrame(records).set_index("Player")

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)


@dataclass(frozen=True)
class WelfarePayoffMatrixResult:
    """Container for 2-player or multi-player strategic trade war normal-form payoff matrix.

    Attributes
    ----------
    players : tuple[str, ...]
        Tuple of two strategic players (e.g. ('USA', 'CHN')).
    strategies : tuple[str, ...]
        Tuple of strategies for player 1 and 2 (e.g. ('Cooperate (0%)', 'Defect (Nash / Optimal Tariff)')).
    payoff_matrix : np.ndarray, shape (2, 2, 2)
        Payoff matrix where payoff_matrix[i, j, k] is the percentage welfare change
        of player k when player 1 plays strategy i and player 2 plays strategy j.
    scenarios : dict[str, TradeEquilibriumResult]
        Equilibrium states keyed by outcome ('CC', 'CD', 'DC', 'DD').
    summary_df : pd.DataFrame
        Formatted 2x2 normal-form payoff matrix DataFrame.
    welfare_metric : str, default 'geary_khamis'
        Welfare metric used to compute payoffs.
    metadata : dict[str, Any], default empty
        Additional diagnostics, tariff rates, and scenario information.
    """

    players: tuple[str, ...]
    strategies: tuple[str, ...]
    payoff_matrix: np.ndarray
    scenarios: dict[str, TradeEquilibriumResult]
    summary_df: pd.DataFrame
    welfare_metric: str = "geary_khamis"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_prisoners_dilemma(self) -> bool:
        """Check whether the payoff matrix satisfies the strict Prisoner's Dilemma inequalities.

        Specifically for both players:
        - Defection is strictly dominant: Payoff(D, C) > Payoff(C, C) and Payoff(D, D) > Payoff(C, D)
        - Mutual cooperation Pareto dominates mutual defection: Payoff(C, C) > Payoff(D, D).
        """
        if self.payoff_matrix.shape != (2, 2, 2):
            return False
        # Player 0: rows are actions (0=C, 1=D)
        p0_dc_gt_cc = self.payoff_matrix[1, 0, 0] > self.payoff_matrix[0, 0, 0]
        p0_dd_gt_cd = self.payoff_matrix[1, 1, 0] > self.payoff_matrix[0, 1, 0]
        p0_cc_gt_dd = self.payoff_matrix[0, 0, 0] > self.payoff_matrix[1, 1, 0]

        # Player 1: cols are actions (0=C, 1=D)
        p1_cd_gt_cc = self.payoff_matrix[0, 1, 1] > self.payoff_matrix[0, 0, 1]
        p1_dd_gt_dc = self.payoff_matrix[1, 1, 1] > self.payoff_matrix[1, 0, 1]
        p1_cc_gt_dd = self.payoff_matrix[0, 0, 1] > self.payoff_matrix[1, 1, 1]

        return bool(
            p0_dc_gt_cc and p0_dd_gt_cd and p0_cc_gt_dd
            and p1_cd_gt_cc and p1_dd_gt_dc and p1_cc_gt_dd
        )

    def summary(self) -> pd.DataFrame:
        """Produce summary DataFrame of normal-form payoff matrix."""
        return self.summary_df

    def to_dataframe(self) -> pd.DataFrame:
        """Return summary table as a pandas DataFrame."""
        return self.summary()

    def to_markdown(self, **kwargs: Any) -> str:
        """Render summary table as GitHub-flavored Markdown."""
        return df_to_markdown(self.summary(), **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        """Render summary table as LaTeX tabular."""
        return df_to_latex(self.summary(), **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        """Render summary table as Typst table."""
        return df_to_typst(self.summary(), **kwargs)


__all__ = [
    "TradeCalibrationResult",
    "TradeEquilibriumResult",
    "GearyKhamisResult",
    "ScenarioBatchResult",
    "RetaliationGameResult",
    "JCurveDynamicResult",
    "RevenueRecyclingResult",
    "CapacityBottleneckResult",
    "OptimalTariffResult",
    "NashTariffResult",
    "WelfarePayoffMatrixResult",
]

