"""Frozen-dataclass result containers for :mod:`puremacro.trade`."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.solvers.diagnostics import SolverDiagnosticRecord, make_solver_diagnostic_record


@dataclass(frozen=True)
class TradeCalibrationResult:
    """Calibrated parameters and baseline endowments for the CGE trade model.

    Stores the output of :func:`calibrate_trade_model`, encapsulating Input-Output
    technical coefficients, final demand sourcing coefficients, Cobb-Douglas
    technology parameters, factor endowments, production taxes, net foreign
    transfers, and baseline gross outputs.

    Attributes
    ----------
    a : np.ndarray, shape (ns*nc, ns, nc)
        Bilateral intermediate input-output technical coefficients:
        ``a[i, j, k] = intermediate_use / gross_output``.
    afd : np.ndarray, shape (ns*nc, nfd, nc)
        Bilateral final demand expenditure coefficients:
        ``afd[i, ifd, k] = final_use / total_final_use``.
    alpha : np.ndarray, shape (1, ns, nc)
        Capital share in value added (Cobb-Douglas exponent).
    beta : np.ndarray, shape (1, ns, nc)
        Total factor productivity scale parameter in value-added production.
    k_endow : np.ndarray, shape (1, nc)
        Total national capital endowments by country (also accessible as ``KT``).
    l_endow : np.ndarray, shape (1, nc)
        Total national labor endowments by country (also accessible as ``LT``).
    invforT : np.ndarray, shape (1, nc)
        Baseline net foreign balance (net exports) by country.
    tax : np.ndarray, shape (1, ns, nc)
        Baseline net production tax rates.
    ytot : np.ndarray, shape (1, ns, nc)
        Baseline gross output quantities (also accessible as ``ytot_base``).
    theta : np.ndarray | None, shape (1, nfd, nc), default None
        Final demand expenditure shares across consumption categories.
    tax_fd : np.ndarray | None, shape (1, nfd, nc), default None
        Final demand tax rates across categories.
    TT : np.ndarray | None, shape (1, nc), default None
        Intermediate production taxes aggregated by country.
    TTfd : np.ndarray | None, shape (1, nc), default None
        Final demand taxes aggregated by country.
    T : np.ndarray | None, shape (1, 1, nc), default None
        Total government tax transfers in baseline level.
    a_3d : np.ndarray | None, shape (ns*nc, ns, nc), default None
        3D intermediate technical coefficients (identical to ``a`` when 3D).
    afd_3d : np.ndarray | None, shape (ns*nc, nfd, nc), default None
        3D final demand sourcing coefficients (identical to ``afd`` when 3D).
    a_4d : np.ndarray | None, shape (ns, nc, ns, nc), default None
        4D intermediate technical coefficient tensor.
    afd_4d : np.ndarray | None, shape (ns, nc, nfd, nc), default None
        4D final demand sourcing coefficient tensor.
    data_calibra : np.ndarray | None, shape (ns*nc + 3, ns*nc + nfd*nc), default None
        Reconstructed benchmark transaction matrix matching original data.
    n_countries : int | None, default None
        Number of countries in the calibrated system; read from ``ytot`` (shape
        ``(1, ns, nc)``) when omitted.
    n_sectors : int | None, default None
        Number of sectors in the calibrated system; read from ``ytot`` when omitted.
    n_final_demand : int, default 3
        Number of final demand categories.
    country_codes : tuple[str, ...], default ()
        Country identifier codes.
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
    n_countries: int = 0
    n_sectors: int = 0
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

        # Dimensions default to the shape of the output array, (1, ns, nc).
        if not self.n_countries or not self.n_sectors:
            if self.ytot.ndim != 3 or self.ytot.shape[0] != 1:
                raise ValueError(
                    "TradeCalibrationResult reads n_sectors and n_countries from 'ytot', which must "
                    f"have shape (1, ns, nc); got {self.ytot.shape}. Pass n_countries and n_sectors."
                )
            if not self.n_sectors:
                object.__setattr__(self, "n_sectors", int(self.ytot.shape[1]))
            if not self.n_countries:
                object.__setattr__(self, "n_countries", int(self.ytot.shape[2]))

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

    # Short aliases
    @property
    def KT(self) -> np.ndarray:
        """Alias for ``k_endow`` (total capital by country)."""
        return self.k_endow

    @property
    def LT(self) -> np.ndarray:
        """Alias for ``l_endow`` (total labour by country)."""
        return self.l_endow

    @property
    def ytot_base(self) -> np.ndarray:
        """Alias for ``ytot`` (baseline gross output)."""
        return self.ytot

    @property
    def nc(self) -> int:
        """Alias for ``n_countries``."""
        return self.n_countries

    @property
    def ns(self) -> int:
        """Alias for ``n_sectors``."""
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
        # category 1 (investment) includes changes in inventories, which are
        # negative when inventories are drawn down. The other categories must be
        # non-negative, and the sourcing shares of an active category sum to 1.
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
    x_sol : np.ndarray, shape (2*ns*nc + 4*nc - 1,)
        Full state vector solution: [log(p); log(y); log(r); log(w); T; XN].
    p_sol : np.ndarray, shape (1, ns, nc)
        Equilibrium gross output prices.
    y_sol : np.ndarray, shape (1, ns, nc)
        Equilibrium gross output quantities.
    r_sol : np.ndarray, shape (1, 1, nc) or (1, nc)
        Equilibrium capital rental rates.
    w_sol : np.ndarray, shape (1, 1, nc) or (1, nc)
        Equilibrium wage rates.
    T_sol : np.ndarray, shape (1, 1, nc) or (1, nc)
        Equilibrium government tax revenues.
    XN_sol : np.ndarray, shape (nc - 1,) or (nc - 1, 1)
        Equilibrium net exports of every country but the last (the last closes the balance).
    intermediate_flows : np.ndarray | None, shape (ns, nc, ns, nc), default None
        4D bilateral intermediate transaction flows tensor.
    final_demand_flows : np.ndarray | None, shape (ns, nc, nfd, nc), default None
        4D bilateral final demand deliveries tensor.
    p_fd : np.ndarray | None, shape (1, nfd, nc), default None
        Composite final demand purchaser prices before consumption taxes.
    c_fd : np.ndarray | None, shape (1, nfd, nc), default None
        Composite final demand real absorption volumes.
    gdp : np.ndarray | None, shape (nc,), default None
        National GDP at market prices (factor cost plus net taxes and tariffs).
    gdp_fc : np.ndarray | None, shape (nc,), default None
        National GDP at factor cost (labor income plus capital income).
    imports : np.ndarray | None, shape (nc,), default None
        National total gross imports across intermediate and final goods.
    exports : np.ndarray | None, shape (nc,), default None
        National total gross exports across intermediate and final goods.
    tariffs : np.ndarray | None, shape (nc,), default None
        National total tariff revenue collections.
    cpi : np.ndarray | None, shape (nc,), default None
        Domestic wage-deflated consumer price index relative to baseline.
    terms_of_trade : np.ndarray | None, shape (nc,), default None
        National terms of trade index (export price index / import price index).
    c_sol : np.ndarray | None, shape (1, nfd, nc), default None
        Alias for c_fd.
    pfd_sol : np.ndarray | None, shape (1, nfd, nc), default None
        Alias for p_fd.
    Pfd_final : np.ndarray | None, shape (1, nfd, nc), default None
        Final demand tax-inclusive consumer prices.
    qxX0_sol : np.ndarray | None, shape (ns, ns, nc, nc), default None
        Bilateral intermediate trade flow quantities, axes (origin sector, destination
        sector, origin country, destination country).
    qxFD0_sol : np.ndarray | None, shape (ns, nfd, nc, nc), default None
        Bilateral final demand trade flow quantities, axes (origin sector, final-use
        category, origin country, destination country).
    data_model_vf : np.ndarray | None, shape (ns*nc + 3, ns*nc + nfd*nc), default None
        Reconstructed equilibrium transaction matrix.
    data_tariff_vf : np.ndarray | None, shape (ns*nc + 4, ns*nc + nfd*nc), default None
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
        Country identifier codes.
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
    solver_diagnostics: SolverDiagnosticRecord | None = None

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

        if self.solver_diagnostics is None:
            conv_val = bool(self.converged)
            iters = int(self.iterations)
            stopping = str(self.metadata.get("solver_termination", "converged" if conv_val else "failed"))
            method = str(self.metadata.get("method", "equilibrated_newton"))
            tol = float(self.metadata.get("tol", 1e-5))

            norm_val = float(self.diff if self.diff != 0.0 else self.residual_norm)
            max_res = float(self.max_residual if self.max_residual != 0.0 else norm_val)
            acc_tol = float(self.metadata.get("accuracy_tolerance", 1e-4))
            is_acc = bool(conv_val and max_res <= acc_tol and np.isfinite(max_res))

            p_min = float(np.min(self.p_sol)) if self.p_sol.size else 0.0
            w_min = float(np.min(self.w_sol)) if self.w_sol.size else 0.0
            r_min = float(np.min(self.r_sol)) if self.r_sol.size else 0.0
            y_min = float(np.min(self.y_sol)) if self.y_sol.size else 0.0

            supp_warnings: list[str] = []
            is_supp = True
            regime = "interior"

            if p_min <= 0.0 or w_min <= 0.0 or r_min <= 0.0:
                is_supp = False
                regime = "infeasible_prices"
                supp_warnings.append(
                    f"Non-positive prices or factor returns (p_min={p_min:.3e}, w_min={w_min:.3e}, r_min={r_min:.3e})"
                )
            if y_min < 0.0:
                is_supp = False
                regime = "negative_gross_output"
                supp_warnings.append(f"Negative gross output detected (y_min={y_min:.3e})")

            cond = self.metadata.get("jacobian_condition")
            if cond is not None and (not np.isfinite(cond) or float(cond) > 1e12):
                supp_warnings.append(f"Ill-conditioned Jacobian (condition={float(cond):.3e})")

            boundary_metrics: dict[str, Any] = {
                "min_price": p_min,
                "min_wage": w_min,
                "min_rental": r_min,
                "min_output": y_min,
            }
            if cond is not None:
                boundary_metrics["jacobian_condition"] = float(cond)

            rec = make_solver_diagnostic_record(
                converged=conv_val,
                iterations=iters,
                stopping_reason=stopping,
                solver_method=method,
                tolerance=tol,
                is_accurate=is_acc,
                max_residual=max_res,
                residual_norm=norm_val,
                residual_type="excess_demand",
                accuracy_tolerance=acc_tol,
                residuals={"diff": norm_val, "max_residual": max_res},
                is_supported=is_supp,
                regime=regime,
                warnings=supp_warnings,
                boundary_metrics=boundary_metrics,
                evidence=dict(self.metadata),
            )
            object.__setattr__(self, "solver_diagnostics", rec)

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
            diag_status = (
                self.solver_diagnostics.overall_status
                if self.solver_diagnostics is not None
                else ("CONVERGED" if self.converged else "FAILED")
            )
            records = [
                {"Diagnostic": "Solver Status", "Value": "CONVERGED" if self.converged else "FAILED"},
                {"Diagnostic": "Overall Status", "Value": diag_status},
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
            else (len(self.country_codes) if self.country_codes else self._n_countries())
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

    def _n_countries(self) -> int:
        """Number of countries, read from the last axis of the price array."""
        return int(self.p_sol.shape[-1]) if self.p_sol.ndim >= 2 else int(self.p_sol.size)

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
            else (len(self.country_codes) if self.country_codes else self._n_countries())
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
    pi : np.ndarray, shape (nfd,)
        International reference commodity prices.
    ppp : np.ndarray, shape (nc,)
        Purchasing power parity exchange rates per country.
    real_gdp : np.ndarray, shape (nc,)
        Real GDP in Geary-Khamis international dollars.
    nominal_gdp : np.ndarray, shape (nc,)
        Nominal expenditure in domestic currency / base units.
    world_real_gdp : float
        Sum of real GDP across all countries.
    world_nominal_gdp : float
        Sum of nominal GDP across all countries.
    gdp_growth : np.ndarray | None, shape (nc,), default None
        Percentage real GDP growth vs. the baseline: the counterfactual and
        baseline final-demand quantities both valued at this solution's
        international prices ``pi``, so it does not depend on the numeraire.
    country_codes : tuple[str, ...], default ()
        Country identifier codes.
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
    real_gdp_growth : np.ndarray | None, shape (nc,), default None
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
        agg_data = {
            "Metric": [
                "World_Nominal_GDP",
                "World_Real_GDP_GK",
                "Mean_PPP",
                "Converged",
                "Iterations",
                "Tolerance",
                "Method",
            ],
            "Value": [
                f"{self.world_nominal_gdp:,.2f}",
                f"{self.world_real_gdp:,.2f}",
                f"{float(np.mean(self.ppp)):.4f}",
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


def _full_net_exports(xn: np.ndarray, nc: int) -> np.ndarray:
    """Net exports of all ``nc`` countries from the equilibrium vector ``XN``.

    ``XN`` holds the first ``nc - 1`` countries; the last one closes the world
    balance, so it is minus their sum. A full-length vector is returned as is.
    """
    xn_flat = np.asarray(xn, dtype=float).ravel()
    if len(xn_flat) == nc:
        return xn_flat.copy()
    xn_full = np.zeros(nc, dtype=float)
    k = min(len(xn_flat), nc - 1)
    xn_full[:k] = xn_flat[:k]
    if len(xn_flat) == nc - 1:
        xn_full[nc - 1] = -float(np.sum(xn_flat))
    return xn_full


def _restore_xn(c: np.ndarray, xn: np.ndarray, nc: int) -> np.ndarray:
    """Add each country's net exports back into the final-use category that carries
    the foreign balance (1, investment; 0 when there is a single category)."""
    c_arr = np.asarray(c, dtype=float).copy()
    xn_full = _full_net_exports(xn, nc)
    if c_arr.ndim in (2, 3):
        inv = 1 if c_arr.shape[-2] >= 2 else 0
        c_arr[..., inv, :] += xn_full
    return c_arr


@dataclass(frozen=True)
class ScenarioBatchResult:
    """Container for comparative multi-scenario counterfactual evaluations.

    Attributes
    ----------
    scenarios : dict[str, TradeEquilibriumResult]
        Equilibrium results keyed by scenario name (e.g. 'base', 'tariff_10').
    geary_khamis : dict[str, GearyKhamisResult]
        Geary-Khamis multilateral PPP results keyed by scenario name.
    baseline_scenario : str, default 'base'
        Identifier of the baseline benchmark scenario.
    country_codes : tuple[str, ...], default ()
        Tuple of country codes (taken from the first result when empty).
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
            nc = len(self.country_codes) if self.country_codes else self._n_countries()
            c_idx = (
                pd.Index(list(self.country_codes), name="Country")
                if self.country_codes
                else pd.Index([f"C{i:02d}" for i in range(nc)], name="Country")
            )

            gdp_mat = np.zeros((nc, len(cols)), dtype=float)
            cpi_mat = np.zeros((nc, len(cols)), dtype=float)
            tb_mat = np.zeros((nc, len(cols)), dtype=float)

            base_res = self.scenarios.get(self.baseline_scenario)
            base_gdp = None
            if base_res is not None and base_res.c_sol is not None:
                c_b = _restore_xn(base_res.c_sol, base_res.XN_sol, nc)
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
                    c_s = _restore_xn(res.c_sol, res.XN_sol, nc)
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
                    tb_mat[:, j] = _full_net_exports(res.XN_sol, nc)

            object.__setattr__(self, "real_gdp_table", pd.DataFrame(gdp_mat, index=c_idx, columns=cols))
            object.__setattr__(self, "cpi_table", pd.DataFrame(cpi_mat, index=c_idx, columns=cols))
            object.__setattr__(self, "trade_balance_table", pd.DataFrame(tb_mat, index=c_idx, columns=cols))

    def _n_countries(self) -> int:
        """Number of countries, read from the first equilibrium result."""
        first = next(iter(self.scenarios.values()), None)
        if first is not None:
            return first._n_countries()
        first_gk = next(iter(self.geary_khamis.values()))
        return len(first_gk.ppp)

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
            if base_res is not None and base_res.c_sol is not None:
                c_b = _restore_xn(base_res.c_sol, base_res.XN_sol, nc)
                base_gdp = np.sum(c_b[0] if c_b.ndim == 3 else c_b, axis=0)

            for s_name, res in self.scenarios.items():
                if res.XN_sol is not None:
                    xn_full = _full_net_exports(res.XN_sol, nc)

                    if s_name == self.baseline_scenario and base_gdp is not None:
                        gdp_val = base_gdp
                    elif res.pfd_sol is not None and res.w_sol is not None and res.c_sol is not None:
                        c_s = _restore_xn(res.c_sol, res.XN_sol, nc)
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

    def summary(self, metric: str = "real_gdp_growth") -> pd.DataFrame:
        """Countries by scenarios for one metric (see :meth:`to_frame`)."""
        return self.to_frame(metric)

    def to_markdown(self, metric: str = "real_gdp_growth", **kwargs: Any) -> str:
        """Render the ``metric`` table as GitHub-flavored Markdown."""
        return df_to_markdown(self.to_frame(metric), **kwargs)

    def to_latex(self, metric: str = "real_gdp_growth", **kwargs: Any) -> str:
        """Render the ``metric`` table as LaTeX tabular."""
        return df_to_latex(self.to_frame(metric), **kwargs)

    def to_typst(self, metric: str = "real_gdp_growth", **kwargs: Any) -> str:
        """Render the ``metric`` table as a Typst table."""
        return df_to_typst(self.to_frame(metric), **kwargs)


@dataclass(frozen=True)
class OptimalTariffResult:
    """Result container for unilateral terms-of-trade optimal tariff optimization.

    Attributes
    ----------
    country_code : str
        Country setting the optimal tariff (e.g. 'C01').
    optimal_tariff_rate : float
        Optimal ad-valorem tariff rate (e.g. 0.2 for 20%).
    welfare_gain_pct : float
        For hicksian_ev: 100 * EV / baseline selected consumption expenditure.
        Other metrics use (W^* - W_0) / W_0 * 100.
    baseline_welfare : float
        Baseline welfare value; zero for hicksian_ev.
    optimal_welfare : float
        Equilibrium welfare under optimal tariff; monetary EV for hicksian_ev.
    welfare_metric : str
        Welfare objective, including explicit 'hicksian_ev' or historical metrics.
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
        Tuple of strategic player country/bloc codes (e.g. ('C01', 'C02', 'C03')).
    nash_tariffs : dict[str, float]
        Equilibrium ad-valorem tariff rates chosen by each player in the Nash equilibrium.
    welfare_changes_pct : dict[str, float]
        Percentage welfare change for each strategic player relative to Free Trade.
        For hicksian_ev, 100 * EV / baseline selected consumption expenditure.
    terms_of_trade_changes_pct : dict[str, float]
        Percentage terms of trade change for each player relative to Free Trade.
    world_welfare_change_pct : float
        Percentage change in aggregate global real welfare. For hicksian_ev,
        summed EV / summed baseline consumption * 100, including nonplayers.
    outer_iterations : int
        Number of outer policy iterations executed.
    converged : bool
        Whether policy iteration converged within tolerance.
    outer_error : float
        Final simultaneous undamped best-response tariff gap (infinity norm).
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
        Tuple of two strategic players (e.g. ('C01', 'C02')).
    strategies : tuple[str, ...]
        Tuple of strategies for player 1 and 2 (e.g. ('Cooperate (0%)', 'Defect (Nash / Optimal Tariff)')).
    payoff_matrix : np.ndarray, shape (2, 2, 2)
        Payoff matrix where payoff_matrix[i, j, k] is the percentage welfare change
        of player k when player 1 plays strategy i and player 2 plays strategy j.
        For hicksian_ev, percentages use baseline selected consumption and each
        player has the same two fixed actions in every cell.
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
    "OptimalTariffResult",
    "NashTariffResult",
    "WelfarePayoffMatrixResult",
]
