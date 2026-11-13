"""Frozen-dataclass result objects for puremacro.var.* estimators."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from puremacro.solvers.diagnostics import SolverDiagnosticRecord, make_solver_diagnostic_record


@dataclass(frozen=True)
class VarEstimateResult:
    """Result of :func:`puremacro.var.estimate.estimate_var`.

    Attributes
    ----------
    A_list : list of ndarray, length p
        VAR coefficient matrices A_1, ..., A_p, each shape (n, n).
    c : ndarray, shape (n,)
        Constant term.
    Sigma : ndarray, shape (n, n)
        Reduced-form residual covariance.
    resid : ndarray, shape (T - p, n)
        Reduced-form residuals.
    X : ndarray, shape (T - p, 1 + n*p)
        Design matrix (constant + p lags).
    """

    A_list: list[np.ndarray]
    c: np.ndarray
    Sigma: np.ndarray
    resid: np.ndarray
    X: np.ndarray
    names: tuple[str, ...] = ()
    solver_diagnostics: SolverDiagnosticRecord | None = None

    def __post_init__(self) -> None:
        if self.solver_diagnostics is None:
            p = len(self.A_list)
            n = self.Sigma.shape[0] if self.Sigma.ndim >= 2 else (len(self.c) if self.c.size else 0)
            rho = 0.0
            if p > 0 and n > 0:
                comp = np.zeros((n * p, n * p), dtype=float)
                for j in range(p):
                    comp[:n, j * n : (j + 1) * n] = self.A_list[j]
                if p > 1:
                    comp[n:, : n * (p - 1)] = np.eye(n * (p - 1))
                try:
                    eigvals = np.linalg.eigvals(comp)
                    rho = float(np.max(np.abs(eigvals)))
                except Exception:
                    rho = float("nan")

            is_supp = bool(np.isfinite(rho) and rho < 1.0)
            regime = "stationary" if is_supp else "non_stationary_explosive"
            supp_warnings: list[str] = []
            if not is_supp:
                supp_warnings.append(
                    f"Explosive VAR root detected: companion matrix spectral radius rho={rho:.4f} >= 1.0. "
                    "Regime is unsupported for stationary impulse responses."
                )

            max_res = float(np.max(np.abs(self.resid))) if self.resid.size else 0.0
            norm_res = float(np.linalg.norm(self.resid)) if self.resid.size else 0.0

            rec = make_solver_diagnostic_record(
                converged=True,
                iterations=1,
                stopping_reason="closed_form_ols",
                solver_method="ols",
                tolerance=0.0,
                is_accurate=bool(np.isfinite(max_res)),
                max_residual=max_res,
                residual_norm=norm_res,
                residual_type="ols_residuals",
                accuracy_tolerance=0.0,
                residuals={"max_residual": max_res, "residual_norm": norm_res},
                is_supported=is_supp,
                regime=regime,
                warnings=supp_warnings,
                boundary_metrics={"spectral_radius": rho, "p": p, "n": n},
            )
            object.__setattr__(self, "solver_diagnostics", rec)

    @property
    def spectral_radius(self) -> float:
        """Companion matrix spectral radius rho(Comp(A))."""
        if self.solver_diagnostics is not None:
            return float(self.solver_diagnostics.support.boundary_metrics.get("spectral_radius", 0.0))
        return 0.0

    def __iter__(self):
        """Support legacy `A_list, c, Sigma, resid, X = estimate_var(...)` unpack."""
        yield self.A_list
        yield self.c
        yield self.Sigma
        yield self.resid
        yield self.X

    def __len__(self):
        """Return 5 so that `len(result) == 5` guards in existing tests pass."""
        return 5

    def __getitem__(self, index: int):
        """Support legacy `result[i]` indexing: 0=A_list, 1=c, 2=Sigma, 3=resid, 4=X."""
        fields = (self.A_list, self.c, self.Sigma, self.resid, self.X)
        return fields[index]

    def summary(self) -> str:
        p = len(self.A_list)
        n = self.Sigma.shape[0]
        T_eff = self.resid.shape[0]
        diag_status = (
            self.solver_diagnostics.overall_status
            if self.solver_diagnostics is not None
            else "PASS"
        )
        return (
            f"VAR estimate\n"
            f"  variables (n)     : {n}\n"
            f"  lag order (p)     : {p}\n"
            f"  effective T       : {T_eff}\n"
            f"  Σ trace           : {float(np.trace(self.Sigma)):.4f}\n"
            f"  spectral radius   : {self.spectral_radius:.4f}\n"
            f"  status            : {diag_status}\n"
        )

    def irf(self, horizon: int = 20, B0: np.ndarray | None = None) -> np.ndarray:
        """Compute orthogonalised impulse responses of shape (horizon+1, n, n).

        If B0 is not provided, defaults to the lower-triangular Cholesky factor of Sigma.
        """
        from .irf import irf as compute_irf
        from .._linalg import safe_cholesky

        if B0 is None:
            B0 = safe_cholesky(self.Sigma, name="VarEstimateResult.irf")
        return compute_irf(self.A_list, B0, horizon=horizon)

    def fevd(self, horizon: int = 20, B0: np.ndarray | None = None) -> np.ndarray:
        """Compute forecast-error variance decomposition of shape (horizon+1, n, n)."""
        from .irf import fevd as compute_fevd
        from .._linalg import safe_cholesky

        if B0 is None:
            B0 = safe_cholesky(self.Sigma, name="VarEstimateResult.fevd")
        return compute_fevd(self.A_list, B0, horizon=horizon)

    def plot(
        self,
        target: int | str = 0,
        shock: int | str = 0,
        horizon: int = 20,
        B0: np.ndarray | None = None,
        title: str = "",
        ylabel: str = "Response",
        scale: float = 1.0,
        ax=None,
    ):
        """Plot impulse response of target variable to shock."""
        from ..plot import plot_irf_single

        target_idx: int = (
            self.names.index(target) if target in self.names else int(target)
        ) if isinstance(target, str) else int(target)
        shock_idx: int = (
            self.names.index(shock) if shock in self.names else int(shock)
        ) if isinstance(shock, str) else int(shock)

        irf_arr = self.irf(horizon=horizon, B0=B0)
        if not title:
            y_name = self.names[target_idx] if target_idx < len(self.names) else f"y_{target_idx}"
            s_name = self.names[shock_idx] if shock_idx < len(self.names) else f"shock_{shock_idx}"
            title = f"Response of {y_name} to {s_name}"

        return plot_irf_single(
            {"irf_point": irf_arr, "names": list(self.names)},
            target_idx=target_idx,
            shock_idx=shock_idx,
            title=title,
            ylabel=ylabel,
            scale=scale,
            ax=ax,
        )

    def to_frame(
        self,
        target: int | str = 0,
        shock: int | str = 0,
        horizon: int = 20,
        B0: np.ndarray | None = None,
    ):
        """Return a tidy DataFrame of the IRF path for target variable to shock."""
        import pandas as pd

        target_idx = self.names.index(target) if isinstance(target, str) and target in self.names else int(target)
        shock_idx = self.names.index(shock) if isinstance(shock, str) and shock in self.names else int(shock)
        irf_arr = self.irf(horizon=horizon, B0=B0)
        return pd.DataFrame({
            "h": np.arange(horizon + 1),
            "response": irf_arr[:, target_idx, shock_idx],
        })

    def to_markdown(
        self,
        target: int | str = 0,
        shock: int | str = 0,
        horizon: int = 20,
        B0: np.ndarray | None = None,
        **kwargs,
    ) -> str:
        """Render IRF table as Markdown."""
        from ..reports import _df_to_markdown

        return _df_to_markdown(
            self.to_frame(target=target, shock=shock, horizon=horizon, B0=B0), **kwargs
        )

    def to_latex(
        self,
        target: int | str = 0,
        shock: int | str = 0,
        horizon: int = 20,
        B0: np.ndarray | None = None,
        **kwargs,
    ) -> str:
        """Render IRF table as LaTeX tabular."""
        from ..reports import _df_to_latex

        return _df_to_latex(
            self.to_frame(target=target, shock=shock, horizon=horizon, B0=B0), **kwargs
        )

    def to_typst(
        self,
        target: int | str = 0,
        shock: int | str = 0,
        horizon: int = 20,
        B0: np.ndarray | None = None,
        **kwargs,
    ) -> str:
        """Render IRF table as Typst table."""
        from ..reports import _df_to_typst

        return _df_to_typst(
            self.to_frame(target=target, shock=shock, horizon=horizon, B0=B0), **kwargs
        )
