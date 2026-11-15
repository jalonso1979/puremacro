"""Empirical Distributional LP to One-Asset HANK Household Incidence Bridge.

Connects the financed sequence-space HANK solver
(:mod:`puremacro.models.hank_fiscal`) and one-asset household block
(:mod:`puremacro.models.hank_sequence_space`) to empirical distributional and
quantile local projections (:mod:`puremacro.lp.distributional`).

Core capabilities
-----------------
1. **Cross-sectional group extraction & simulation**: Computes exact
   consumption and net-asset impulse responses across :math:`G` household groups
   (liquid-wealth quintiles/deciles/quartiles/terciles, borrowing-constrained
   hand-to-mouth vs. unconstrained households, income states, or custom state
   partitions) from solved transitions under ``balanced_budget``,
   ``debt_rule``, ``proportional``, ``lump_sum``, monetary, or cash-on-hand
   transfer shocks.
2. **Fail-closed conservation auditing**: Verifies that the population-weighted
   sum of group consumption IRFs equals aggregate ``dC_t`` to tight tolerance
   (default ``< 1e-10``), that consumption share changes sum to ``0``, and that
   group net-asset IRFs sum to aggregate ``dA_t`` without silent rescaling.
3. **Empirical alignment & scoring**: Aligns model group trajectories against
   :class:`~puremacro.lp.distributional.DistributionalLPResult` or observed
   group targets and standard errors, reporting per-group RMSE/MAE, sign
   concordance, pointwise t-statistics and confidence-band coverage, cross-group
   inequality-gradient contrasts, steady-state MPC / HtM calibration metadata,
   and a standardized :class:`~puremacro.solvers.diagnostics.SolverDiagnosticRecord`.
4. **Bilingual reporting**: Supports ``lang="en" | "es"`` across
   ``.summary()``, ``.to_markdown()``, and ``.to_dict()``.

Validation scope
----------------
Evaluates deterministic perfect-foresight MIT transitions or first-order
sequence-space perturbations around the stationary one-asset household
distribution. It does not solve higher-order aggregate-risk equilibria with
stochastic risk premia or two-asset illiquid-liquid portfolio rebalancing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro.solvers.diagnostics import (
    AccuracyStatus,
    ConvergenceStatus,
    SolverDiagnosticRecord,
    SupportStatus,
    make_solver_diagnostic_record,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "EmpiricalGroupTargets",
    "HANKDistributionalBridgeResult",
    "HANKGroupIRFResult",
    "bridge_hank_distributional",
    "extract_hank_group_irf",
    "score_hank_distributional",
    "simulate_transfer_group_irf",
    "solve_hank_group_irf",
]

_DEFAULT_CONSERVATION_TOL = 1e-10
_SIGN_ZERO_TOL = 1e-12
_HTM_ASSET_TOL = 1e-8
_TOP_GRID_MASS_WARN = 1e-6


# ---------------------------------------------------------------------------
# Internal validation, formatting, and JSON-serialization helpers
# ---------------------------------------------------------------------------


def _resolve_lang(lang: str) -> str:
    """Validate and normalize bilingual language selector ('en' or 'es')."""
    if not isinstance(lang, str):
        raise TypeError(f"lang must be a string ('en' or 'es'), got {type(lang).__name__}")
    norm = lang.strip().lower()
    if norm not in ("en", "es"):
        raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
    return norm


def _json_sanitize(obj: Any) -> Any:
    """Convert numpy/pandas objects recursively to plain JSON-serializable values."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_json_sanitize(x) for x in obj.tolist()]
    if isinstance(obj, pd.Series):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [_json_sanitize(rec) for rec in obj.to_dict(orient="records")]
    if isinstance(obj, Mapping):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


def _as_finite_array(values: Any, name: str, *, ndim: int | None = None) -> np.ndarray:
    """Convert array-like input to float64 ndarray and fail closed on NaN/Inf/complex."""
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    arr = np.asarray(values, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional, got shape {arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} cannot be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite (NaN or Inf) entries")
    return arr


def _safe_ratio(num: float, den: float, *, zero_tol: float = 1e-12) -> float:
    """Return num / den when finite and |den| > zero_tol, else NaN."""
    if not (math.isfinite(num) and math.isfinite(den)) or abs(den) <= zero_tol:
        return float("nan")
    return float(num / den)


def _sign_match(a: float, b: float, *, zero_tol: float = _SIGN_ZERO_TOL) -> bool:
    """Directional sign agreement with consistent near-zero handling."""
    if abs(a) <= zero_tol and abs(b) <= zero_tol:
        return True
    if abs(a) <= zero_tol or abs(b) <= zero_tol:
        return False
    return bool((a > 0.0) == (b > 0.0))


def _scope_note(scope: str, lang: str) -> str:
    """Return explicit validation-scope disclosure in English or Spanish."""
    if lang == "es":
        if scope == "linearized_sequence_space":
            return (
                "Alcance de validación: respuesta linealizada de primer orden en espacio de "
                "secuencias alrededor de la distribución estacionaria de un activo; no incluye "
                "riesgo agregado no lineal ni portafolios de dos activos."
            )
        if scope == "partial_equilibrium_transfer":
            return (
                "Alcance de validación: transferencia en equilibrio parcial sobre la política "
                "estacionaria del bloque de hogares de un activo (sin retroalimentación de tasas "
                "ni precios de equilibrio general)."
            )
        return (
            "Alcance de validación: transición determinista MIT con previsión perfecta en un "
            "modelo HANK de un activo alrededor de la distribución estacionaria; no modela "
            "riesgo agregado estocástico ni recomposición entre activos líquidos e ilíquidos."
        )
    if scope == "linearized_sequence_space":
        return (
            "Validation scope: first-order linearized sequence-space perturbation around the "
            "stationary one-asset distribution; excludes nonlinear aggregate risk and two-asset "
            "illiquid-liquid portfolio rebalancing."
        )
    if scope == "partial_equilibrium_transfer":
        return (
            "Validation scope: partial-equilibrium cash-on-hand transfer propagated through the "
            "stationary one-asset household block (fixed steady-state prices and interest rate)."
        )
    return (
        "Validation scope: deterministic perfect-foresight MIT transition around the stationary "
        "one-asset distribution; excludes stochastic aggregate risk premia and two-asset "
        "illiquid-liquid portfolio rebalancing."
    )


def _norm_label(label: str) -> str:
    """Normalize group/bin label for exact case-insensitive lookup."""
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


# ---------------------------------------------------------------------------
# Public result and target containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EmpiricalGroupTargets:
    """Validated container for observed or LP-estimated group impulse responses.

    Attributes
    ----------
    horizons : np.ndarray
        Integer impulse-response horizons of shape ``(H,)``.
    groups : tuple[str, ...]
        Ordered names of the ``G`` cross-sectional household groups.
    irf_matrix : np.ndarray
        Observed group impulse responses of shape ``(H, G)``.
    se_matrix : np.ndarray or None, optional
        Pointwise standard errors of shape ``(H, G)``.
    ci_lower, ci_upper : np.ndarray or None, optional
        Lower and upper confidence bounds of shape ``(H, G)``.
    ci_level : float, default 0.90
        Nominal confidence coverage in ``(0, 1)``.
    distribution_type : str, default 'levels'
        Target scale tag (``'levels'``, ``'shares'``, ``'percent'``, ``'quantiles'``,
        or ``'assets'``).
    shock_name : str, default ''
        Descriptive name of the empirical shock.
    metadata : dict[str, Any]
        Optional provenance or estimation metadata.
    """

    horizons: np.ndarray
    groups: tuple[str, ...]
    irf_matrix: np.ndarray
    se_matrix: np.ndarray | None = None
    ci_lower: np.ndarray | None = None
    ci_upper: np.ndarray | None = None
    ci_level: float = 0.90
    distribution_type: str = "levels"
    shock_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        h_raw = np.asarray(self.horizons)
        if h_raw.ndim != 1 or h_raw.size == 0:
            raise ValueError("EmpiricalGroupTargets.horizons must be a non-empty 1D integer array")
        if np.issubdtype(h_raw.dtype, np.floating) and not np.all(np.isfinite(h_raw)):
            raise ValueError("EmpiricalGroupTargets.horizons contains non-finite values")
        h_int = h_raw.astype(np.int64)
        if np.any(h_int.astype(np.float64) != h_raw.astype(np.float64)):
            raise ValueError("EmpiricalGroupTargets.horizons must contain integer horizons")
        if len(np.unique(h_int)) != len(h_int):
            raise ValueError("EmpiricalGroupTargets.horizons must be unique")
        object.__setattr__(self, "horizons", h_int)

        grp = tuple(str(g) for g in self.groups)
        if len(grp) < 2:
            raise ValueError(f"EmpiricalGroupTargets requires at least 2 groups, got {len(grp)}")
        if any(not g.strip() for g in grp) or len(set(grp)) != len(grp):
            raise ValueError(f"EmpiricalGroupTargets.groups must be unique non-empty names, got {grp!r}")
        object.__setattr__(self, "groups", grp)

        irf = _as_finite_array(self.irf_matrix, "EmpiricalGroupTargets.irf_matrix", ndim=2)
        expected_shape = (len(h_int), len(grp))
        if irf.shape != expected_shape:
            raise ValueError(
                f"EmpiricalGroupTargets.irf_matrix has shape {irf.shape}, expected {expected_shape}"
            )
        object.__setattr__(self, "irf_matrix", irf)

        ci_lvl = float(self.ci_level)
        if not (math.isfinite(ci_lvl) and 0.0 < ci_lvl < 1.0):
            raise ValueError(f"EmpiricalGroupTargets.ci_level must be in (0, 1), got {ci_lvl!r}")
        object.__setattr__(self, "ci_level", ci_lvl)

        se_arr: np.ndarray | None = None
        if self.se_matrix is not None:
            se_arr = _as_finite_array(self.se_matrix, "EmpiricalGroupTargets.se_matrix", ndim=2)
            if se_arr.shape != expected_shape:
                raise ValueError(
                    f"EmpiricalGroupTargets.se_matrix has shape {se_arr.shape}, expected {expected_shape}"
                )
            if np.any(se_arr < 0.0):
                raise ValueError("EmpiricalGroupTargets.se_matrix cannot contain negative values")
        object.__setattr__(self, "se_matrix", se_arr)

        lo_arr: np.ndarray | None = None
        hi_arr: np.ndarray | None = None
        if self.ci_lower is not None or self.ci_upper is not None:
            if self.ci_lower is None or self.ci_upper is None:
                raise ValueError("Both ci_lower and ci_upper must be provided together")
            lo_arr = _as_finite_array(self.ci_lower, "EmpiricalGroupTargets.ci_lower", ndim=2)
            hi_arr = _as_finite_array(self.ci_upper, "EmpiricalGroupTargets.ci_upper", ndim=2)
            if lo_arr.shape != expected_shape or hi_arr.shape != expected_shape:
                raise ValueError(
                    f"EmpiricalGroupTargets CI bounds must have shape {expected_shape}, "
                    f"got {lo_arr.shape} and {hi_arr.shape}"
                )
            if np.any(lo_arr > hi_arr):
                raise ValueError("EmpiricalGroupTargets.ci_lower exceeds ci_upper")
        elif se_arr is not None:
            from scipy.stats import norm

            z = float(norm.ppf(1.0 - (1.0 - ci_lvl) / 2.0))
            lo_arr = irf - z * se_arr
            hi_arr = irf + z * se_arr

        object.__setattr__(self, "ci_lower", lo_arr)
        object.__setattr__(self, "ci_upper", hi_arr)


@dataclass(frozen=True)
class HANKGroupIRFResult:
    """Cross-sectional household-group consumption and net-asset IRFs from HANK.

    Attributes
    ----------
    horizons : np.ndarray
        Integer horizons ``0, ..., T - 1`` of shape ``(T,)``.
    groups : tuple[str, ...]
        Ordered household group labels of length ``G``.
    group_scheme : str
        Grouping specification name (e.g. ``'quintiles'``, ``'deciles'``, ``'htm'``).
    tracking : str
        Dynamic tracking convention: ``'cohort'``, ``'cross_section'``, or ``'fixed_bins'``.
    model_kind : str
        Underlying HANK model type (``'fiscal_nonlinear'``, ``'sequence_space_nonlinear'``,
        ``'sequence_space_linear'``, or ``'partial_equilibrium_transfer'``).
    closure : str
        Equilibrium fiscal/asset closure description.
    tax_type : str
        Household tax/transfer incidence rule (``'proportional'``, ``'lump_sum'``, etc.).
    consumption_irf : np.ndarray
        Per-capita group consumption level deviations ``dc_{t, g}`` of shape ``(T, G)``.
    consumption_pct_irf : np.ndarray
        Relative group consumption deviations ``dc_{t, g} / c_{g, ss}`` of shape ``(T, G)``.
    consumption_share_irf : np.ndarray
        Group share changes in aggregate consumption ``ds_{t, g}^C`` of shape ``(T, G)``.
    asset_irf : np.ndarray
        Per-capita group net-asset deviations ``da_{t, g}`` of shape ``(T, G)``.
    asset_share_irf : np.ndarray
        Group share changes in aggregate net assets ``ds_{t, g}^A`` of shape ``(T, G)``.
    aggregate_consumption_irf : np.ndarray
        Aggregate consumption IRF ``dC_t`` from the underlying HANK model of shape ``(T,)``.
    aggregate_asset_irf : np.ndarray
        Aggregate household net-asset IRF ``dA_t`` of shape ``(T,)``.
    group_masses : np.ndarray
        Steady-state population share ``p_g`` of each group of shape ``(G,)``.
    group_masses_path : np.ndarray
        Group population shares over ``t = 0, ..., T - 1`` of shape ``(T, G)``.
    steady_state_group_consumption : np.ndarray
        Steady-state per-capita group consumption ``c_{g, ss}`` of shape ``(G,)``.
    steady_state_group_assets : np.ndarray
        Steady-state per-capita group net assets ``a_{g, ss}`` of shape ``(G,)``.
    steady_state_group_mpc : np.ndarray
        Steady-state group average quarterly MPC of shape ``(G,)``.
    steady_state_group_consumption_shares : np.ndarray
        Steady-state consumption share ``s_{g, ss}^C`` of shape ``(G,)``.
    conservation_audit : dict[str, float]
        Exact cross-sectional adding-up and accounting conservation residuals.
    calibration : dict[str, Any]
        Explicit steady-state calibration metadata (aggregate/group MPCs, HtM shares, Gini).
    solver_diagnostics : SolverDiagnosticRecord
        Standardized convergence, accuracy, and support status record.
    validation_scope : str
        Identifier for the model validation boundary.
    source_result : Any
        Underlying HANK solver result object.
    """

    horizons: np.ndarray
    groups: tuple[str, ...]
    group_scheme: str
    tracking: str
    model_kind: str
    closure: str
    tax_type: str
    consumption_irf: np.ndarray
    consumption_pct_irf: np.ndarray
    consumption_share_irf: np.ndarray
    asset_irf: np.ndarray
    asset_share_irf: np.ndarray
    aggregate_consumption_irf: np.ndarray
    aggregate_asset_irf: np.ndarray
    group_masses: np.ndarray
    group_masses_path: np.ndarray
    steady_state_group_consumption: np.ndarray
    steady_state_group_assets: np.ndarray
    steady_state_group_mpc: np.ndarray
    steady_state_group_consumption_shares: np.ndarray
    conservation_audit: dict[str, float]
    calibration: dict[str, Any]
    solver_diagnostics: SolverDiagnosticRecord
    validation_scope: str = "deterministic_mit_transition"
    source_result: Any = field(default=None, repr=False)

    @property
    def convergence_status(self) -> ConvergenceStatus:
        """Numerical convergence indicators."""
        return self.solver_diagnostics.convergence

    @property
    def accuracy_status(self) -> AccuracyStatus:
        """Accounting conservation and market-clearing accuracy indicators."""
        return self.solver_diagnostics.accuracy

    @property
    def support_status(self) -> SupportStatus:
        """Grid support and economic regime indicators."""
        return self.solver_diagnostics.support

    def _select_matrix(self, metric: str) -> tuple[np.ndarray, str, np.ndarray | None]:
        key = str(metric).strip().lower()
        if key in ("consumption", "consumption_level", "level", "c"):
            return self.consumption_irf, "Aggregate_dC", self.aggregate_consumption_irf
        if key in ("consumption_pct", "pct", "percent", "c_pct"):
            c_ss = float(self.calibration["C_ss"])
            agg_pct = self.aggregate_consumption_irf / c_ss if abs(c_ss) > 1e-12 else None
            return self.consumption_pct_irf, "Aggregate_dC_pct", agg_pct
        if key in ("consumption_share", "share", "shares", "c_share"):
            return self.consumption_share_irf, "Share_Sum", np.sum(self.consumption_share_irf, axis=1)
        if key in ("asset", "assets", "asset_level", "a"):
            return self.asset_irf, "Aggregate_dA", self.aggregate_asset_irf
        if key in ("asset_share", "a_share"):
            return self.asset_share_irf, "Asset_Share_Sum", np.sum(self.asset_share_irf, axis=1)
        raise ValueError(
            f"Unknown metric {metric!r}; expected 'consumption', 'consumption_pct', "
            f"'consumption_share', 'asset', or 'asset_share'."
        )

    def to_frame(self, metric: str = "consumption", *, long: bool = False) -> pd.DataFrame:
        """Return group IRF paths as a wide or long pandas DataFrame."""
        if long:
            rows: list[dict[str, Any]] = []
            for t_idx, h in enumerate(self.horizons):
                for g_idx, g_name in enumerate(self.groups):
                    rows.append(
                        {
                            "horizon": int(h),
                            "group": g_name,
                            "consumption_irf": float(self.consumption_irf[t_idx, g_idx]),
                            "consumption_pct_irf": float(self.consumption_pct_irf[t_idx, g_idx]),
                            "consumption_share_irf": float(self.consumption_share_irf[t_idx, g_idx]),
                            "asset_irf": float(self.asset_irf[t_idx, g_idx]),
                            "asset_share_irf": float(self.asset_share_irf[t_idx, g_idx]),
                            "population_mass": float(self.group_masses_path[t_idx, g_idx]),
                        }
                    )
            return pd.DataFrame(rows)

        mat, agg_col, agg_vec = self._select_matrix(metric)
        data: dict[str, Any] = {g: mat[:, idx] for idx, g in enumerate(self.groups)}
        if agg_vec is not None:
            data[agg_col] = agg_vec
        return pd.DataFrame(data, index=pd.Index(self.horizons, name="horizon"))

    def to_dataframe(self, metric: str = "consumption", *, long: bool = False) -> pd.DataFrame:
        """Alias for :meth:`to_frame`."""
        return self.to_frame(metric=metric, long=long)

    def summary(self, lang: str = "en") -> str:
        """Render a bilingual ('en' or 'es') digest of group IRFs and conservation checks."""
        use_lang = _resolve_lang(lang)
        ca = self.conservation_audit
        cal = self.calibration
        status = self.solver_diagnostics.overall_status
        c0 = self.consumption_irf[0]
        low_lbl, high_lbl = self.groups[0], self.groups[-1]
        grad0 = float(c0[0] - c0[-1])
        mpc0, mpc_last = float(self.steady_state_group_mpc[0]), float(self.steady_state_group_mpc[-1])

        if use_lang == "es":
            lines = [
                f"IRF Distribucional de Hogares HANK [{status}]",
                "=" * 72,
                f"Modelo / Cierre / Impuesto   : {self.model_kind} / {self.closure} / {self.tax_type}",
                f"Grupos / Seguimiento / T     : {self.group_scheme} (G={len(self.groups)}) / {self.tracking} / {len(self.horizons)} trimestres",
                f"MPC agregado / HtM restr.    : {cal['aggregate_mpc']:.4f} / {cal['htm_constrained_share']:.4f} (Gini riqueza={cal['wealth_gini']:.4f})",
                f"MPC ({low_lbl} vs {high_lbl})       : {mpc0:.4f} vs {mpc_last:.4f}",
                f"Impacto t=0 ({low_lbl} - {high_lbl}): {grad0:+.6f} (dC_0={self.aggregate_consumption_irf[0]:+.6f})",
                f"Conservación dC / dShare     : {ca['consumption_adding_up_error']:.3e} / {ca['consumption_share_adding_up_error']:.3e}",
                f"Conservación dA / Masa       : {ca['asset_adding_up_error']:.3e} / {ca['population_mass_error']:.3e}",
                _scope_note(self.validation_scope, "es"),
                "=" * 72,
            ]
            return "\n".join(lines)

        lines = [
            f"HANK Cross-Sectional Household Group IRF [{status}]",
            "=" * 72,
            f"Model / Closure / Tax        : {self.model_kind} / {self.closure} / {self.tax_type}",
            f"Groups / Tracking / Horizon  : {self.group_scheme} (G={len(self.groups)}) / {self.tracking} / {len(self.horizons)} quarters",
            f"Aggregate MPC / HtM Share    : {cal['aggregate_mpc']:.4f} / {cal['htm_constrained_share']:.4f} (wealth Gini={cal['wealth_gini']:.4f})",
            f"Group MPC ({low_lbl} vs {high_lbl})  : {mpc0:.4f} vs {mpc_last:.4f}",
            f"Impact t=0 ({low_lbl} - {high_lbl}): {grad0:+.6f} (dC_0={self.aggregate_consumption_irf[0]:+.6f})",
            f"Conservation dC / dShare err : {ca['consumption_adding_up_error']:.3e} / {ca['consumption_share_adding_up_error']:.3e}",
            f"Conservation dA / Mass err   : {ca['asset_adding_up_error']:.3e} / {ca['population_mass_error']:.3e}",
            _scope_note(self.validation_scope, "en"),
            "=" * 72,
        ]
        return "\n".join(lines)

    def to_markdown(self, lang: str = "en", metric: str = "consumption", **kwargs: Any) -> str:
        """Render bilingual Markdown report with summary, calibration, and group IRF table."""
        from puremacro.reports import _df_to_markdown

        use_lang = _resolve_lang(lang)
        df_ss = pd.DataFrame(
            {
                "mass": self.group_masses,
                "mpc_ss": self.steady_state_group_mpc,
                "wealth_ss": self.steady_state_group_assets,
                "cons_ss": self.steady_state_group_consumption,
                "dc_impact": self.consumption_irf[0],
                "dshare_impact": self.consumption_share_irf[0],
                "da_impact": self.asset_irf[0],
            },
            index=pd.Index(self.groups, name="group"),
        )
        if use_lang == "es":
            df_ss.columns = [
                "masa_pob",
                "mpc_ee",
                "riqueza_ee",
                "consumo_ee",
                "dc_impacto",
                "dcuota_impacto",
                "da_impacto",
            ]
            heading = "# Resumen Distribucional de Hogares HANK"
            sub1 = "## Calibración estacionaria e impacto en t=0"
            sub2 = f"## Trayectoria por grupos ({metric})"
        else:
            heading = "# HANK Cross-Sectional Household Incidence"
            sub1 = "## Steady-State Calibration and Impact (t=0)"
            sub2 = f"## Group Impulse Response Paths ({metric})"

        parts = [
            heading,
            "",
            self.summary(lang=use_lang),
            "",
            sub1,
            "",
            _df_to_markdown(df_ss, **kwargs),
            "",
            sub2,
            "",
            _df_to_markdown(self.to_frame(metric=metric).head(12), **kwargs),
        ]
        return "\n".join(parts)

    def to_latex(self, metric: str = "consumption", **kwargs: Any) -> str:
        """Render group IRF paths as a LaTeX tabular environment."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(metric=metric), **kwargs)

    def to_typst(self, metric: str = "consumption", **kwargs: Any) -> str:
        """Render group IRF paths as Typst table markup."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(metric=metric), **kwargs)

    def to_dict(self, lang: str = "en") -> dict[str, Any]:
        """Serialize group IRFs, conservation audits, and calibration to JSON-safe dict."""
        use_lang = _resolve_lang(lang)
        return {
            "lang": use_lang,
            "summary": self.summary(lang=use_lang),
            "summaries": {
                "en": self.summary(lang="en"),
                "es": self.summary(lang="es"),
            },
            "scope_note": _scope_note(self.validation_scope, use_lang),
            "validation_scope": self.validation_scope,
            "model_kind": self.model_kind,
            "closure": self.closure,
            "tax_type": self.tax_type,
            "group_scheme": self.group_scheme,
            "tracking": self.tracking,
            "horizons": _json_sanitize(self.horizons),
            "groups": list(self.groups),
            "group_masses": _json_sanitize(self.group_masses),
            "consumption_irf": _json_sanitize(self.consumption_irf),
            "consumption_pct_irf": _json_sanitize(self.consumption_pct_irf),
            "consumption_share_irf": _json_sanitize(self.consumption_share_irf),
            "asset_irf": _json_sanitize(self.asset_irf),
            "asset_share_irf": _json_sanitize(self.asset_share_irf),
            "aggregate_consumption_irf": _json_sanitize(self.aggregate_consumption_irf),
            "aggregate_asset_irf": _json_sanitize(self.aggregate_asset_irf),
            "conservation_audit": _json_sanitize(self.conservation_audit),
            "calibration": _json_sanitize(self.calibration),
            "solver_diagnostics": self.solver_diagnostics.to_dict(),
        }

    def plot(
        self,
        metric: str = "consumption",
        *,
        style: str = "publication",
        figsize: tuple[float, float] = (11.5, 4.5),
    ) -> Figure:
        """Plot cross-sectional group IRF paths and steady-state MPC / impact incidence."""
        import matplotlib.pyplot as plt

        from puremacro.plot import _palette

        mat, _, agg_vec = self._select_matrix(metric)
        G = len(self.groups)
        colors = _palette(G) if style == "grayscale" else [
            plt.cm.viridis(i / max(1, G - 1)) for i in range(G)
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        for idx, g_name in enumerate(self.groups):
            ax1.plot(self.horizons, mat[:, idx], label=g_name, color=colors[idx], lw=1.8)
        if agg_vec is not None and metric not in ("consumption_share", "asset_share"):
            ax1.plot(self.horizons, agg_vec, color="black", ls="--", lw=1.5, label="Aggregate")
        ax1.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax1.set_title(f"HANK Group IRF ({metric})", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Horizon (quarters)", fontsize=9)
        ax1.set_ylabel("Deviation from SS", fontsize=9)
        ax1.legend(frameon=False, ncol=2 if G > 5 else 1, fontsize=8)
        ax1.grid(True, ls=":", alpha=0.4)

        x = np.arange(G)
        width = 0.36
        ax2.bar(x - width / 2, self.steady_state_group_mpc, width, label="SS MPC", color="#2b5c8f")
        scale_imp = float(np.max(np.abs(mat[0])))
        imp_norm = mat[0] / scale_imp * float(np.max(self.steady_state_group_mpc)) if scale_imp > 1e-15 else mat[0]
        ax2.bar(x + width / 2, imp_norm, width, label="Scaled t=0 IRF", color="#c0392b", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.groups, rotation=30 if G > 5 else 0, ha="right" if G > 5 else "center", fontsize=8)
        ax2.set_title("Steady-State MPC & Impact Incidence", fontsize=10, fontweight="bold")
        ax2.legend(frameon=False, fontsize=8)
        ax2.grid(True, ls=":", alpha=0.4)

        fig.tight_layout()
        return fig


@dataclass(frozen=True)
class HANKDistributionalBridgeResult:
    """Empirical Distributional LP vs. HANK household-group alignment scorecard.

    Attributes
    ----------
    group_irf : HANKGroupIRFResult
        Underlying conservation-audited HANK cross-sectional group IRF result.
    aligned_horizons : np.ndarray
        Aligned integer horizons ``(H_al,)``.
    groups : tuple[str, ...]
        Aligned model group names ``(G_al,)``.
    empirical_bins : tuple[str, ...]
        Aligned empirical bin names ``(G_al,)``.
    response_type : str
        Aligned response unit (``'level'``, ``'percent'``, ``'share'``,
        ``'asset_level'``, or ``'asset_share'``).
    shock_scale : float
        Explicit scale factor applied to model group IRFs before comparison.
    model_irf, empirical_irf, empirical_se, ci_lower, ci_upper : np.ndarray
        Aligned arrays of shape ``(H_al, G_al)``.
    ci_level : float
        Nominal empirical confidence level.
    aligned_table : pd.DataFrame
        Pointwise aligned comparison table including t-stats and band coverage.
    per_group_metrics : pd.DataFrame
        Per-group discrepancy, concordance, coverage, timing, and steady-state MPCs.
    gradient_metrics : dict[str, Any]
        Cross-group dispersion and inequality-gradient contrast metrics.
    gradient_table : pd.DataFrame
        Horizon-by-horizon low-vs-high group contrast and cross-group dispersion.
    rmse, mae, weighted_rmse : float
        Overall discrepancy metrics across aligned ``(horizon, group)`` entries.
    chi2_stat, chi2_df, chi2_pvalue : float / int
        Diagonal Wald statistic using empirical pointwise variances.
    sign_concordance : float
        Share of aligned ``(horizon, group)`` points with matching sign.
    band_coverage : float
        Share of aligned points where the model IRF falls inside empirical bands.
    mean_abs_t_stat, max_abs_t_stat : float
        Mean and maximum absolute pointwise t-statistic ``|(model - emp) / se|``.
    mean_trajectory_correlation : float
        Mean per-group temporal correlation across aligned horizons.
    calibration : dict[str, Any]
        Steady-state MPC and HtM calibration metadata.
    conservation_audit : dict[str, float]
        Cross-sectional conservation audit residuals from ``group_irf``.
    solver_diagnostics : SolverDiagnosticRecord
        Standardized convergence, accuracy, and support diagnostic record.
    warnings : tuple[str, ...]
        Diagnostic, alignment, and scope warnings.
    validation_scope : str
        Scope identifier of the structural HANK evaluation.
    """

    group_irf: HANKGroupIRFResult
    aligned_horizons: np.ndarray
    groups: tuple[str, ...]
    empirical_bins: tuple[str, ...]
    response_type: str
    shock_scale: float
    model_irf: np.ndarray
    empirical_irf: np.ndarray
    empirical_se: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    ci_level: float
    aligned_table: pd.DataFrame
    per_group_metrics: pd.DataFrame
    gradient_metrics: dict[str, Any]
    gradient_table: pd.DataFrame
    rmse: float
    mae: float
    weighted_rmse: float
    chi2_stat: float
    chi2_df: int
    chi2_pvalue: float
    sign_concordance: float
    band_coverage: float
    mean_abs_t_stat: float
    max_abs_t_stat: float
    mean_trajectory_correlation: float
    calibration: dict[str, Any]
    conservation_audit: dict[str, float]
    solver_diagnostics: SolverDiagnosticRecord
    warnings: tuple[str, ...] = ()
    validation_scope: str = "deterministic_mit_transition"

    @property
    def convergence_status(self) -> ConvergenceStatus:
        """Numerical convergence indicators."""
        return self.solver_diagnostics.convergence

    @property
    def accuracy_status(self) -> AccuracyStatus:
        """Accounting conservation and market-clearing accuracy indicators."""
        return self.solver_diagnostics.accuracy

    @property
    def support_status(self) -> SupportStatus:
        """Grid support and economic regime indicators."""
        return self.solver_diagnostics.support

    def to_frame(
        self,
        layout: Literal["per_group", "aligned", "gradient", "metrics"] = "per_group",
    ) -> pd.DataFrame:
        """Return per-group scorecard, pointwise aligned table, gradient table, or summary metrics."""
        if layout == "per_group":
            return self.per_group_metrics.copy()
        if layout == "aligned":
            return self.aligned_table.copy()
        if layout == "gradient":
            return self.gradient_table.copy()
        if layout == "metrics":
            gm = self.gradient_metrics
            rows = [
                {"metric": "rmse", "value": self.rmse},
                {"metric": "mae", "value": self.mae},
                {"metric": "weighted_rmse", "value": self.weighted_rmse},
                {"metric": "chi2_stat", "value": self.chi2_stat},
                {"metric": "chi2_df", "value": float(self.chi2_df)},
                {"metric": "chi2_pvalue", "value": self.chi2_pvalue},
                {"metric": "sign_concordance", "value": self.sign_concordance},
                {"metric": "band_coverage", "value": self.band_coverage},
                {"metric": "mean_abs_t_stat", "value": self.mean_abs_t_stat},
                {"metric": "max_abs_t_stat", "value": self.max_abs_t_stat},
                {"metric": "mean_trajectory_correlation", "value": self.mean_trajectory_correlation},
                {"metric": "gradient_diff_rmse", "value": float(gm["gradient_diff_rmse"])},
                {"metric": "gradient_sign_concordance", "value": float(gm["gradient_sign_concordance"])},
                {"metric": "dispersion_ratio", "value": float(gm["dispersion_ratio"])},
                {"metric": "aggregate_mpc", "value": float(self.calibration["aggregate_mpc"])},
                {"metric": "htm_constrained_share", "value": float(self.calibration["htm_constrained_share"])},
            ]
            return pd.DataFrame(rows).set_index("metric")
        raise ValueError(
            f"Unknown layout {layout!r}; expected 'per_group', 'aligned', 'gradient', or 'metrics'."
        )

    def to_dataframe(
        self,
        layout: Literal["per_group", "aligned", "gradient", "metrics"] = "per_group",
    ) -> pd.DataFrame:
        """Alias for :meth:`to_frame`."""
        return self.to_frame(layout=layout)

    def summary(self, lang: str = "en") -> str:
        """Render a bilingual ('en' or 'es') empirical-vs-HANK distributional scorecard."""
        use_lang = _resolve_lang(lang)
        status = self.solver_diagnostics.overall_status
        gm = self.gradient_metrics
        cal = self.calibration
        ca = self.conservation_audit
        h_span = f"[{int(self.aligned_horizons[0])}..{int(self.aligned_horizons[-1])}]"
        wald = (
            f"chi2({self.chi2_df})={self.chi2_stat:.3f}, p={self.chi2_pvalue:.4g}"
            if self.chi2_df > 0 and math.isfinite(self.chi2_stat) and math.isfinite(self.chi2_pvalue)
            else "n/a"
        )
        cov_txt = f"{self.band_coverage:.3f}" if math.isfinite(self.band_coverage) else "n/a"
        t_txt = f"{self.mean_abs_t_stat:.3f}" if math.isfinite(self.mean_abs_t_stat) else "n/a"

        if use_lang == "es":
            lines = [
                f"Puente Distribucional LP Empírica <-> HANK [{status}]",
                "=" * 74,
                f"Modelo / Cierre / Escala     : {self.group_irf.model_kind} ({self.group_irf.closure}, {self.group_irf.tax_type}) | unidad={self.response_type} (x{self.shock_scale:g})",
                f"Horizontes / Grupos alineados: {h_span} (H={len(self.aligned_horizons)}) | G={len(self.groups)} ({gm['low_group']} vs {gm['high_group']})",
                f"Ajuste global RMSE / MAE     : {self.rmse:.6f} / {self.mae:.6f} (W-RMSE={self.weighted_rmse:.3f}, {wald})",
                f"Signo / Cobertura IC / |t|   : {self.sign_concordance:.3f} / {cov_txt} / {t_txt}",
                f"Gradiente ({gm['low_group']}-{gm['high_group']}): impacto mod={gm['gradient_impact_diff_model']:+.6f} vs emp={gm['gradient_impact_diff_empirical']:+.6f} (RMSE={gm['gradient_diff_rmse']:.6f}, signo={gm['gradient_sign_concordance']:.3f})",
                f"Calibración MPC / HtM / Gini : {cal['aggregate_mpc']:.4f} / {cal['htm_constrained_share']:.4f} / {cal['wealth_gini']:.4f}",
                f"Conservación dC / dShare     : {ca['consumption_adding_up_error']:.3e} / {ca['consumption_share_adding_up_error']:.3e}",
                _scope_note(self.validation_scope, "es"),
                "=" * 74,
            ]
            return "\n".join(lines)

        lines = [
            f"Empirical Distributional LP <-> HANK Incidence Bridge [{status}]",
            "=" * 74,
            f"Model / Closure / Scale      : {self.group_irf.model_kind} ({self.group_irf.closure}, {self.group_irf.tax_type}) | unit={self.response_type} (x{self.shock_scale:g})",
            f"Aligned Horizons / Groups    : {h_span} (H={len(self.aligned_horizons)}) | G={len(self.groups)} ({gm['low_group']} vs {gm['high_group']})",
            f"Overall Fit RMSE / MAE       : {self.rmse:.6f} / {self.mae:.6f} (W-RMSE={self.weighted_rmse:.3f}, {wald})",
            f"Sign / Band Cov. / Mean |t|  : {self.sign_concordance:.3f} / {cov_txt} / {t_txt}",
            f"Gradient ({gm['low_group']}-{gm['high_group']}): impact mod={gm['gradient_impact_diff_model']:+.6f} vs emp={gm['gradient_impact_diff_empirical']:+.6f} (RMSE={gm['gradient_diff_rmse']:.6f}, sign={gm['gradient_sign_concordance']:.3f})",
            f"Calibration MPC / HtM / Gini : {cal['aggregate_mpc']:.4f} / {cal['htm_constrained_share']:.4f} / {cal['wealth_gini']:.4f}",
            f"Conservation dC / dShare err : {ca['consumption_adding_up_error']:.3e} / {ca['consumption_share_adding_up_error']:.3e}",
            _scope_note(self.validation_scope, "en"),
            "=" * 74,
        ]
        return "\n".join(lines)

    def to_markdown(
        self,
        lang: str = "en",
        layout: Literal["per_group", "aligned", "gradient", "metrics"] = "per_group",
        **kwargs: Any,
    ) -> str:
        """Render bilingual Markdown scorecard with summary, gradient contrast, and tables."""
        from puremacro.reports import _df_to_markdown

        use_lang = _resolve_lang(lang)
        gm = self.gradient_metrics
        grad_df = pd.DataFrame(
            [
                {
                    "metric": "low_minus_high_impact",
                    "model": gm["gradient_impact_diff_model"],
                    "empirical": gm["gradient_impact_diff_empirical"],
                },
                {
                    "metric": "low_minus_high_cumulative",
                    "model": gm["gradient_cumulative_diff_model"],
                    "empirical": gm["gradient_cumulative_diff_empirical"],
                },
                {
                    "metric": "low_to_high_ratio_impact",
                    "model": gm["gradient_impact_ratio_model"],
                    "empirical": gm["gradient_impact_ratio_empirical"],
                },
                {
                    "metric": "low_to_high_ratio_cumulative",
                    "model": gm["gradient_cumulative_ratio_model"],
                    "empirical": gm["gradient_cumulative_ratio_empirical"],
                },
                {
                    "metric": "cross_group_dispersion_mean",
                    "model": gm["dispersion_mean_model"],
                    "empirical": gm["dispersion_mean_empirical"],
                },
            ]
        ).set_index("metric")

        if use_lang == "es":
            grad_df.columns = ["modelo", "empirico"]
            grad_df.index = pd.Index(
                [
                    "dif_bajo_menos_alto_impacto",
                    "dif_bajo_menos_alto_acumulada",
                    "razon_bajo_a_alto_impacto",
                    "razon_bajo_a_alto_acumulada",
                    "dispersion_media_entre_grupos",
                ],
                name="indicador",
            )
            title = "# Puente Empírico LP Distribucional <-> HANK"
            h_grad = f"## Contraste de Gradiente ({gm['low_group']} vs {gm['high_group']})"
            h_tbl = f"## Tabla de Evaluación ({layout})"
        else:
            title = "# Empirical Distributional LP <-> HANK Bridge Scorecard"
            h_grad = f"## Inequality-Gradient Contrast ({gm['low_group']} vs {gm['high_group']})"
            h_tbl = f"## Alignment Table ({layout})"

        parts = [
            title,
            "",
            self.summary(lang=use_lang),
            "",
            h_grad,
            "",
            _df_to_markdown(grad_df, **kwargs),
            "",
            h_tbl,
            "",
            _df_to_markdown(self.to_frame(layout=layout), **kwargs),
        ]
        return "\n".join(parts)

    def to_latex(
        self,
        layout: Literal["per_group", "aligned", "gradient", "metrics"] = "per_group",
        **kwargs: Any,
    ) -> str:
        """Render aligned comparison table as LaTeX tabular markup."""
        from puremacro.reports import _df_to_latex

        return _df_to_latex(self.to_frame(layout=layout), **kwargs)

    def to_typst(
        self,
        layout: Literal["per_group", "aligned", "gradient", "metrics"] = "per_group",
        **kwargs: Any,
    ) -> str:
        """Render aligned comparison table as Typst table markup."""
        from puremacro.reports import _df_to_typst

        return _df_to_typst(self.to_frame(layout=layout), **kwargs)

    def to_dict(self, lang: str = "en") -> dict[str, Any]:
        """Serialize bridge alignment scorecard, gradient metrics, and diagnostics."""
        use_lang = _resolve_lang(lang)
        return {
            "lang": use_lang,
            "summary": self.summary(lang=use_lang),
            "summaries": {
                "en": self.summary(lang="en"),
                "es": self.summary(lang="es"),
            },
            "scope_note": _scope_note(self.validation_scope, use_lang),
            "validation_scope": self.validation_scope,
            "model_kind": self.group_irf.model_kind,
            "closure": self.group_irf.closure,
            "tax_type": self.group_irf.tax_type,
            "response_type": self.response_type,
            "shock_scale": float(self.shock_scale),
            "aligned_horizons": _json_sanitize(self.aligned_horizons),
            "groups": list(self.groups),
            "empirical_bins": list(self.empirical_bins),
            "overall_metrics": {
                "rmse": _json_sanitize(self.rmse),
                "mae": _json_sanitize(self.mae),
                "weighted_rmse": _json_sanitize(self.weighted_rmse),
                "chi2_stat": _json_sanitize(self.chi2_stat),
                "chi2_df": int(self.chi2_df),
                "chi2_pvalue": _json_sanitize(self.chi2_pvalue),
                "sign_concordance": _json_sanitize(self.sign_concordance),
                "band_coverage": _json_sanitize(self.band_coverage),
                "mean_abs_t_stat": _json_sanitize(self.mean_abs_t_stat),
                "max_abs_t_stat": _json_sanitize(self.max_abs_t_stat),
                "mean_trajectory_correlation": _json_sanitize(self.mean_trajectory_correlation),
            },
            "gradient_metrics": _json_sanitize(self.gradient_metrics),
            "per_group_metrics": _json_sanitize(self.per_group_metrics.reset_index()),
            "aligned_table": _json_sanitize(self.aligned_table),
            "gradient_table": _json_sanitize(self.gradient_table),
            "calibration": _json_sanitize(self.calibration),
            "conservation_audit": _json_sanitize(self.conservation_audit),
            "solver_diagnostics": self.solver_diagnostics.to_dict(),
            "warnings": list(self.warnings),
        }

    def plot(
        self,
        groups: Sequence[str] | None = None,
        *,
        style: str = "publication",
        figsize: tuple[float, float] = (12.0, 4.8),
    ) -> Figure:
        """Plot empirical LP bands vs. HANK model group paths and low-high contrast."""
        import matplotlib.pyplot as plt

        if groups is None:
            idx_list = [0] if len(self.groups) == 1 else [0, len(self.groups) - 1]
            if len(self.groups) > 2:
                idx_list = [0, len(self.groups) // 2, len(self.groups) - 1]
        else:
            idx_list = [self.groups.index(str(g)) for g in groups]

        palette = ["#1f2937", "#4b5563", "#9ca3af"] if style == "grayscale" else ["#1d4ed8", "#15803d", "#b91c1c", "#7c3aed"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        for k, g_idx in enumerate(idx_list):
            c = palette[k % len(palette)]
            g_lbl = self.groups[g_idx]
            b_lbl = self.empirical_bins[g_idx]
            emp = self.empirical_irf[:, g_idx]
            mod = self.model_irf[:, g_idx]
            ax1.plot(self.aligned_horizons, emp, color=c, lw=1.8, label=f"Emp: {b_lbl}")
            ax1.plot(self.aligned_horizons, mod, color=c, lw=1.8, ls="--", label=f"HANK: {g_lbl}")
            if np.all(np.isfinite(self.ci_lower[:, g_idx])) and np.all(np.isfinite(self.ci_upper[:, g_idx])):
                ax1.fill_between(
                    self.aligned_horizons,
                    self.ci_lower[:, g_idx],
                    self.ci_upper[:, g_idx],
                    color=c,
                    alpha=0.12,
                )
        ax1.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax1.set_title(f"Group IRFs ({self.response_type})", fontsize=10, fontweight="bold")
        ax1.set_xlabel("Horizon h", fontsize=9)
        ax1.legend(frameon=False, fontsize=8, ncol=2)
        ax1.grid(True, ls=":", alpha=0.4)

        gm = self.gradient_metrics
        ax2.plot(
            self.aligned_horizons,
            self.gradient_table["diff_empirical"],
            color=palette[0],
            lw=2.0,
            label=f"Emp ({gm['low_group']} - {gm['high_group']})",
        )
        ax2.plot(
            self.aligned_horizons,
            self.gradient_table["diff_model"],
            color=palette[-1],
            lw=2.0,
            ls="--",
            label=f"HANK ({gm['low_group']} - {gm['high_group']})",
        )
        ax2.axhline(0.0, color="0.5", ls=":", lw=0.8)
        ax2.set_title("Cross-Group Inequality Gradient", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Horizon h", fontsize=9)
        ax2.legend(frameon=False, fontsize=8)
        ax2.grid(True, ls=":", alpha=0.4)

        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Household group partition & steady-state calibration metadata
# ---------------------------------------------------------------------------


def _wealth_gini(a_grid: np.ndarray, D_a: np.ndarray) -> float:
    """Compute Gini coefficient of stationary liquid wealth distribution."""
    mass = np.maximum(np.asarray(D_a, dtype=np.float64), 0.0)
    tot_mass = float(np.sum(mass))
    if tot_mass <= 0.0:
        return float("nan")
    mass = mass / tot_mass
    a_vals = np.maximum(np.asarray(a_grid, dtype=np.float64), 0.0)
    mean_a = float(np.sum(mass * a_vals))
    if mean_a <= 1e-15:
        return 0.0
    cum_pop = np.cumsum(mass)
    cum_wealth = np.cumsum(mass * a_vals) / mean_a
    cum_pop_prev = np.concatenate(([0.0], cum_pop[:-1]))
    cum_wealth_prev = np.concatenate(([0.0], cum_wealth[:-1]))
    lorenz_area = float(0.5 * np.sum((cum_wealth + cum_wealth_prev) * (cum_pop - cum_pop_prev)))
    return float(max(0.0, min(1.0, 1.0 - 2.0 * lorenz_area)))


def _complete_quantile_weights(W_a: np.ndarray, D_a: np.ndarray) -> np.ndarray:
    """Ensure quantile partition weights sum to 1.0 even on zero-density grid tails."""
    out = np.asarray(W_a, dtype=np.float64).copy()
    tot = float(np.sum(D_a))
    cum = np.cumsum(D_a) / tot if tot > 0.0 else np.linspace(0.0, 1.0, len(D_a))
    col_sum = np.sum(out, axis=0)
    deficit = 1.0 - col_sum
    missing = np.abs(deficit) > 1e-11
    if np.any(missing):
        low_tail = missing & (cum < 0.5)
        high_tail = missing & ~low_tail
        out[0, low_tail] += deficit[low_tail]
        out[-1, high_tail] += deficit[high_tail]
    return np.maximum(out, 0.0)


def _quantile_edges_weights(D_a: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    """Split grid-point mass across arbitrary strictly increasing cumulative-mass edges."""
    from puremacro.models.hank_sequence_space import _mass_window_weights

    raw = np.vstack([_mass_window_weights(D_a, float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)])
    return _complete_quantile_weights(raw, D_a)


def _build_group_partition(
    D_ss: np.ndarray,
    a_ss: np.ndarray,
    a_grid: np.ndarray,
    *,
    groups: str | int | Sequence[float] | Mapping[str, np.ndarray],
    group_labels: Sequence[str] | None = None,
    htm_definition: Literal["constrained", "bottom_quartile"] = "constrained",
) -> tuple[tuple[str, ...], np.ndarray, str, Sequence[float] | None]:
    """Construct validated group membership weights ``W`` of shape ``(G, n_a, n_s)``."""
    from puremacro.models.hank_sequence_space import _decile_weights

    n_a, n_s = D_ss.shape
    D_a = D_ss.sum(axis=1)
    quantile_edges: Sequence[float] | None = None

    if isinstance(groups, Mapping):
        if len(groups) < 2:
            raise ValueError(f"Custom group mapping must define at least 2 groups, got {len(groups)}")
        names = tuple(str(k) for k in groups.keys())
        w_list: list[np.ndarray] = []
        for name, raw_w in groups.items():
            w_arr = _as_finite_array(raw_w, f"groups[{name!r}]")
            if w_arr.shape == (n_a,):
                w_2d = np.repeat(w_arr[:, None], n_s, axis=1)
            elif w_arr.shape == (n_a, n_s):
                w_2d = w_arr.copy()
            else:
                raise ValueError(
                    f"groups[{name!r}] must have shape ({n_a},) or ({n_a}, {n_s}), got {w_arr.shape}"
                )
            if np.any(w_2d < -1e-15):
                raise ValueError(f"groups[{name!r}] contains negative membership weights")
            w_2d = np.maximum(w_2d, 0.0)
            w_list.append(w_2d)
        W = np.stack(w_list, axis=0)
        scheme = "custom"
    elif isinstance(groups, (int, np.integer)) and not isinstance(groups, bool):
        G = int(groups)
        if G < 2:
            raise ValueError(f"groups integer count must be >= 2, got {G}")
        W_a = _complete_quantile_weights(_decile_weights(D_a, n_bins=G), D_a)
        W = np.repeat(W_a[:, :, None], n_s, axis=2)
        if G == 10:
            names = tuple(f"D{d + 1}" for d in range(10))
            scheme = "deciles"
        elif G == 5:
            names = tuple(f"Q{q + 1}" for q in range(5))
            scheme = "quintiles"
        elif G == 4:
            names = tuple(f"Q{q + 1}" for q in range(4))
            scheme = "quartiles"
        elif G == 3:
            names = ("T1", "T2", "T3")
            scheme = "terciles"
        else:
            names = tuple(f"G{g + 1}" for g in range(G))
            scheme = f"quantiles_{G}"
        quantile_edges = np.linspace(0.0, 1.0, G + 1).tolist()
    elif isinstance(groups, str):
        key = _norm_label(groups)
        if key in ("quintiles", "quintile", "5"):
            W_a = _complete_quantile_weights(_decile_weights(D_a, n_bins=5), D_a)
            W = np.repeat(W_a[:, :, None], n_s, axis=2)
            names = ("Q1", "Q2", "Q3", "Q4", "Q5")
            scheme = "quintiles"
            quantile_edges = np.linspace(0.0, 1.0, 6).tolist()
        elif key in ("deciles", "decile", "10"):
            W_a = _complete_quantile_weights(_decile_weights(D_a, n_bins=10), D_a)
            W = np.repeat(W_a[:, :, None], n_s, axis=2)
            names = tuple(f"D{d + 1}" for d in range(10))
            scheme = "deciles"
            quantile_edges = np.linspace(0.0, 1.0, 11).tolist()
        elif key in ("quartiles", "quartile", "4"):
            W_a = _complete_quantile_weights(_decile_weights(D_a, n_bins=4), D_a)
            W = np.repeat(W_a[:, :, None], n_s, axis=2)
            names = ("Q1", "Q2", "Q3", "Q4")
            scheme = "quartiles"
            quantile_edges = np.linspace(0.0, 1.0, 5).tolist()
        elif key in ("terciles", "tercile", "3"):
            W_a = _complete_quantile_weights(_decile_weights(D_a, n_bins=3), D_a)
            W = np.repeat(W_a[:, :, None], n_s, axis=2)
            names = ("T1", "T2", "T3")
            scheme = "terciles"
            quantile_edges = np.linspace(0.0, 1.0, 4).tolist()
        elif key in ("htm", "htm_vs_unconstrained", "hand_to_mouth", "liquidity"):
            if htm_definition == "constrained":
                htm_mask = (a_ss <= float(a_grid[0]) + _HTM_ASSET_TOL).astype(np.float64)
                W = np.stack([htm_mask, 1.0 - htm_mask], axis=0)
                scheme = "htm_vs_unconstrained"
            elif htm_definition == "bottom_quartile":
                W_a = _quantile_edges_weights(D_a, (0.0, 0.25, 1.0))
                W = np.repeat(W_a[:, :, None], n_s, axis=2)
                scheme = "htm_bottom_quartile"
                quantile_edges = [0.0, 0.25, 1.0]
            else:
                raise ValueError(
                    f"Unknown htm_definition {htm_definition!r}; expected 'constrained' or 'bottom_quartile'."
                )
            names = ("HtM", "Unconstrained")
        elif key in ("income", "productivity", "s"):
            W = np.zeros((n_s, n_a, n_s), dtype=np.float64)
            for j in range(n_s):
                W[j, :, j] = 1.0
            names = ("Low_Income", "High_Income") if n_s == 2 else tuple(f"Income_{j + 1}" for j in range(n_s))
            scheme = "income"
        elif key in ("wealth_x_income", "liquidity_x_income", "wealth_income"):
            W_a = _quantile_edges_weights(D_a, (0.0, 0.5, 1.0))
            W = np.zeros((2 * n_s, n_a, n_s), dtype=np.float64)
            w_labels: list[str] = []
            idx_g = 0
            for w_idx, w_lbl in enumerate(("Low_Wealth", "High_Wealth")):
                for j in range(n_s):
                    W[idx_g, :, j] = W_a[w_idx]
                    s_lbl = ("Low_Income" if j == 0 else "High_Income") if n_s == 2 else f"Income_{j + 1}"
                    w_labels.append(f"{w_lbl}/{s_lbl}")
                    idx_g += 1
            names = tuple(w_labels)
            scheme = "wealth_x_income"
        else:
            raise ValueError(
                f"Unknown groups specification {groups!r}. Choose 'quintiles', 'deciles', "
                f"'quartiles', 'terciles', 'htm_vs_unconstrained', 'income', 'wealth_x_income', "
                f"an integer G >= 2, quantile edges, or a Mapping[str, weights]."
            )
    elif isinstance(groups, Sequence):
        edges = [float(x) for x in groups]
        if len(edges) < 3:
            raise ValueError("Quantile edges sequence must have at least 3 endpoints (2 groups)")
        if not np.all(np.isfinite(edges)) or abs(edges[0]) > 1e-12 or abs(edges[-1] - 1.0) > 1e-12:
            raise ValueError(f"Quantile edges must start at 0.0 and end at 1.0, got {edges!r}")
        if any(edges[i + 1] <= edges[i] for i in range(len(edges) - 1)):
            raise ValueError(f"Quantile edges must be strictly increasing, got {edges!r}")
        W_a = _quantile_edges_weights(D_a, edges)
        W = np.repeat(W_a[:, :, None], n_s, axis=2)
        names = tuple(f"G{i + 1}" for i in range(len(edges) - 1))
        scheme = "custom_quantiles"
        quantile_edges = edges
    else:
        raise TypeError(f"Unsupported groups type: {type(groups).__name__}")

    if group_labels is not None:
        custom_names = tuple(str(lbl) for lbl in group_labels)
        if len(custom_names) != W.shape[0]:
            raise ValueError(
                f"group_labels length ({len(custom_names)}) must match number of groups ({W.shape[0]})"
            )
        if len(set(custom_names)) != len(custom_names) or any(not lbl.strip() for lbl in custom_names):
            raise ValueError(f"group_labels must be unique non-empty strings, got {custom_names!r}")
        names = custom_names

    active = D_ss > 1e-15
    sum_w = np.sum(W, axis=0)
    if np.any(active) and float(np.max(np.abs(sum_w[active] - 1.0))) > 1e-10:
        raise ValueError(
            "Group partition weights must sum to 1.0 across groups at every active (a, s) state "
            f"(max partition error={float(np.max(np.abs(sum_w[active] - 1.0))):.3e})."
        )
    masses = np.einsum("gij,ij->g", W, D_ss)
    if np.any(masses <= 1e-12):
        empty = [names[i] for i, m in enumerate(masses) if m <= 1e-12]
        hint = " Use htm_definition='bottom_quartile'." if scheme == "htm_vs_unconstrained" else ""
        raise ValueError(
            f"Group partition produced empty/zero-mass group(s) {empty} (min mass={float(np.min(masses)):.3e}).{hint}"
        )

    return names, W, scheme, quantile_edges


def _compute_steady_state_calibration(
    D_ss: np.ndarray,
    c_ss: np.ndarray,
    a_ss: np.ndarray,
    a_grid: np.ndarray,
    r_ss: float,
    W: np.ndarray,
    groups: tuple[str, ...],
    extra_ss: Mapping[str, Any],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute steady-state group moments and calibration metadata."""
    from puremacro.models.hank_sequence_space import _local_mpc

    mpc_grid = _local_mpc(c_ss, a_grid, r_ss)
    agg_mpc = float(np.sum(D_ss * mpc_grid))
    C_ss = float(np.sum(D_ss * c_ss))
    A_ss = float(np.sum(D_ss * a_grid[:, None]))
    D_a = D_ss.sum(axis=1)

    masses = np.einsum("gij,ij->g", W, D_ss)
    tot_c = np.einsum("gij,ij->g", W, D_ss * c_ss)
    tot_a = np.einsum("gij,ij->g", W, D_ss * a_grid[:, None])
    tot_mpc = np.einsum("gij,ij->g", W, D_ss * mpc_grid)

    mean_c = tot_c / masses
    mean_a = tot_a / masses
    group_mpc = tot_mpc / masses
    c_shares = tot_c / C_ss

    constrained_mask = (a_ss <= float(a_grid[0]) + _HTM_ASSET_TOL).astype(np.float64)
    htm_constrained_share = float(np.sum(D_ss * constrained_mask))
    top_grid_mass = float(np.sum(D_ss[-1, :]))

    cal: dict[str, Any] = {
        "aggregate_mpc": agg_mpc,
        "htm_constrained_share": htm_constrained_share,
        "htm_bottom_quartile_share": 0.25,
        "wealth_gini": _wealth_gini(a_grid, D_a),
        "mpc_gradient_low_minus_high": float(group_mpc[0] - group_mpc[-1]),
        "C_ss": C_ss,
        "A_ss": A_ss,
        "r_ss": float(r_ss),
        "n_a": int(len(a_grid)),
        "a_max": float(a_grid[-1]),
        "top_grid_mass": top_grid_mass,
        "group_masses": {g: float(masses[i]) for i, g in enumerate(groups)},
        "group_mpcs": {g: float(group_mpc[i]) for i, g in enumerate(groups)},
        "group_mean_wealth": {g: float(mean_a[i]) for i, g in enumerate(groups)},
        "group_mean_consumption": {g: float(mean_c[i]) for i, g in enumerate(groups)},
        "group_consumption_shares": {g: float(c_shares[i]) for i, g in enumerate(groups)},
    }
    for k, v in extra_ss.items():
        if k not in cal:
            cal[k] = float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v

    return cal, masses, mean_c, mean_a, group_mpc, c_shares


# ---------------------------------------------------------------------------
# Exact path extractors for FiscalHANKResult, NonlinearHANKResult, SequenceSpaceHANKResult
# ---------------------------------------------------------------------------


def _extract_fiscal_paths(
    res: Any,
    *,
    allow_unconverged: bool,
) -> dict[str, Any]:
    """Reconstruct exact household policy paths from a solved FiscalHANKResult."""
    from puremacro.models.hank_fiscal import _disposable_income_vec, _egm_step_fiscal
    from puremacro.models.hank_sequence_space import _income_process

    if not allow_unconverged and not bool(res.converged):
        status = (
            res.solver_diagnostics.overall_status
            if res.solver_diagnostics is not None
            else "FAILED"
        )
        raise ValueError(
            f"FiscalHANKResult did not converge cleanly (overall_status={status}); "
            "pass allow_unconverged=True to inspect failed/unsupported paths explicitly."
        )

    ss_res = res.steady_state_model
    c_ss = np.asarray(ss_res.policy_c, dtype=np.float64)
    a_ss = np.asarray(ss_res.policy_a, dtype=np.float64)
    D_ss = np.asarray(ss_res.distribution, dtype=np.float64)
    a_grid = np.asarray(ss_res.asset_grid, dtype=np.float64)
    s_grid, pi_s = _income_process(c_ss.shape[1])

    ss_meta = dict(res.steady_state)
    r_ss = float(ss_meta.get("r_ss", ss_res.r_ss))
    w_ss = float(ss_meta.get("w_ss", ss_res.w_ss))
    B_ss = float(ss_meta.get("B_ss", ss_res.government_debt))
    G_ss = float(ss_meta.get("G_ss", 0.0))
    T_ss = float(ss_meta.get("T_ss", r_ss * B_ss + G_ss))
    N_ss = float(ss_meta.get("N_ss", 1.0))
    beta = float(ss_res.beta)
    gamma = float(ss_res.gamma)
    V_ss = (1.0 + r_ss) * (c_ss ** (-gamma))

    T = int(res.horizon)
    rr_seq = np.concatenate(([r_ss], np.asarray(res.r_ex_ante, dtype=np.float64)))
    if res.closure == "balanced_budget":
        T_T = float(rr_seq[T]) * B_ss + G_ss
    else:
        phi_B = float(res.phi_B if res.phi_B is not None else 0.0)
        T_T = T_ss + phi_B * (float(res.B[-1]) - B_ss)

    y_T = _disposable_income_vec(w_ss, T_T, T_ss, N_ss, s_grid, str(res.tax_type))
    V_next, _, _ = _egm_step_fiscal(
        V_ss,
        a_grid,
        s_grid,
        pi_s,
        beta,
        gamma,
        float(rr_seq[T]),
        y_T,
        enforce_feasibility=False,
    )

    n_a, n_s = c_ss.shape
    c_path = np.empty((T, n_a, n_s), dtype=np.float64)
    a_path = np.empty((T, n_a, n_s), dtype=np.float64)
    w_path = np.asarray(res.w, dtype=np.float64)
    T_path = np.asarray(res.T, dtype=np.float64)
    for t in range(T - 1, -1, -1):
        y_t = _disposable_income_vec(
            float(w_path[t]),
            float(T_path[t]),
            T_ss,
            N_ss,
            s_grid,
            str(res.tax_type),
        )
        V_next, c_path[t], a_path[t] = _egm_step_fiscal(
            V_next,
            a_grid,
            s_grid,
            pi_s,
            beta,
            gamma,
            float(rr_seq[t]),
            y_t,
            enforce_feasibility=False,
        )

    return {
        "linear": False,
        "model_kind": "fiscal_nonlinear",
        "closure": str(res.closure),
        "tax_type": str(res.tax_type),
        "validation_scope": "deterministic_mit_transition",
        "horizon": T,
        "a_grid": a_grid,
        "s_grid": s_grid,
        "pi_s": pi_s,
        "r_ss": r_ss,
        "c_ss": c_ss,
        "a_ss": a_ss,
        "D_ss": D_ss,
        "c_path": c_path,
        "a_path": a_path,
        "agg_dC": np.asarray(res.dC, dtype=np.float64),
        "agg_dA": np.asarray(res.A, dtype=np.float64) - float(np.sum(D_ss * a_ss)),
        "agg_dB": np.asarray(res.dB, dtype=np.float64),
        "extra_ss": {
            **ss_meta,
            "beta": beta,
            "gamma": gamma,
            "phi_pi": float(ss_res.phi_pi),
            "kappa": float(ss_res.kappa),
        },
        "source_diag": res.solver_diagnostics,
        "source_converged": bool(res.converged),
        "source_iterations": int(res.iterations),
        "source_tol": float(res.tol),
        "source_method": f"sequence_space_broyden_{res.closure}_{res.tax_type}",
    }


def _extract_nonlinear_ssj_paths(
    res: Any,
    *,
    allow_unconverged: bool,
) -> dict[str, Any]:
    """Reconstruct exact household policy paths from a NonlinearHANKResult."""
    from puremacro.models.hank_sequence_space import (
        _household_block_from_result,
        _household_transition,
    )

    if not allow_unconverged and not bool(res.converged):
        raise ValueError(
            "NonlinearHANKResult did not converge cleanly; pass allow_unconverged=True "
            "to inspect unconverged paths explicitly."
        )

    ss_res = res.steady_state_model
    hh = _household_block_from_result(ss_res)
    T = int(res.horizon)

    audit_ctx = getattr(res, "_audit_context", None) or {}
    paths = audit_ctx.get("paths") if isinstance(audit_ctx, Mapping) else None
    if (
        isinstance(paths, Mapping)
        and "consumption" in paths
        and "assets" in paths
        and np.asarray(paths["consumption"]).shape == (T, *hh.c_ss.shape)
    ):
        c_path = np.asarray(paths["consumption"], dtype=np.float64)
        a_path = np.asarray(paths["assets"], dtype=np.float64)
    else:
        dr = np.asarray(res.irf_rate_nonlinear, dtype=np.float64)
        dY = np.asarray(res.nonlinear_path, dtype=np.float64)
        rr_seq = np.concatenate(([hh.r_ss], hh.r_ss + dr))
        w_seq = np.maximum(hh.w_ss * (1.0 + dY / hh.C_ss), 1e-6)
        _, c_path, a_path = _household_transition(hh, rr_seq, w_seq)

    A_path = np.empty(T, dtype=np.float64)
    from puremacro.models.hank_sequence_space import _forward_step

    D_tmp = hh.D_ss.copy()
    for t in range(T):
        A_path[t] = float(np.sum(D_tmp * a_path[t]))
        D_tmp = _forward_step(D_tmp, a_path[t], hh.a_grid, hh.pi_s)

    A_ss = float(np.sum(hh.D_ss * hh.a_grid[:, None]))
    shock_var = str(getattr(res, "shock_var", "r"))
    closure = "balanced_budget_interest" if shock_var == "r" else "unfinanced_goods_only"
    max_res = float(np.max(np.abs(np.asarray(res.residuals, dtype=np.float64))))
    tol_val = float(getattr(res, "tol", 1e-6))

    return {
        "linear": False,
        "model_kind": "sequence_space_nonlinear",
        "closure": closure,
        "tax_type": "proportional",
        "validation_scope": "deterministic_mit_transition",
        "horizon": T,
        "a_grid": hh.a_grid,
        "s_grid": hh.s_grid,
        "pi_s": hh.pi_s,
        "r_ss": hh.r_ss,
        "c_ss": hh.c_ss,
        "a_ss": hh.a_ss,
        "D_ss": hh.D_ss,
        "c_path": c_path,
        "a_path": a_path,
        "agg_dC": np.asarray(res.irf_consumption_nonlinear, dtype=np.float64),
        "agg_dA": A_path - A_ss,
        "agg_dB": None,
        "extra_ss": {
            "Y_ss": hh.C_ss,
            "C_ss": hh.C_ss,
            "G_ss": 0.0,
            "B_ss": hh.B,
            "w_ss": hh.w_ss,
            "beta": hh.beta,
            "gamma": hh.gamma,
            "phi_pi": float(ss_res.phi_pi),
            "kappa": float(ss_res.kappa),
            "shock_var": shock_var,
            "goods_market_residual": max_res,
        },
        "source_diag": None,
        "source_converged": bool(res.converged and hh.converged),
        "source_iterations": int(res.iterations),
        "source_tol": tol_val,
        "source_method": f"sequence_space_broyden_{shock_var}",
    }


def _fake_news_policy_steps(
    hh: Any,
    shock_input: str,
    T_len: int,
    h: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ABRS step-1 policy and distribution derivatives (dc_0^k, da_0^k, dD_1^k)."""
    from puremacro.models.hank_sequence_space import _egm_step, _forward_step

    args = (hh.a_grid, hh.s_grid, hh.pi_s, hh.beta, hh.gamma)
    w_net_ss = hh.net_wage(hh.r_ss, hh.w_ss)
    base_V, base_c, base_a = _egm_step(hh.V_ss, *args, hh.r_ss, w_net_ss)
    D1_base = _forward_step(hh.D_ss, base_a, hh.a_grid, hh.pi_s)

    r_p = hh.r_ss + (h if shock_input == "r" else 0.0)
    w_p = hh.w_ss + (h if shock_input == "w" else 0.0)
    V_p, c_p, a_p = _egm_step(hh.V_ss, *args, r_p, hh.net_wage(r_p, w_p))

    N = hh.c_ss.size
    dc = np.zeros((T_len, N), dtype=np.float64)
    da = np.zeros((T_len, N), dtype=np.float64)
    dD1 = np.zeros((T_len, N), dtype=np.float64)
    dV = (V_p - base_V) / h
    dc_k = (c_p - base_c) / h
    da_k = (a_p - base_a) / h
    for k in range(T_len):
        if k > 0:
            V_k, c_k, a_k = _egm_step(hh.V_ss + h * dV, *args, hh.r_ss, w_net_ss)
            dV = (V_k - base_V) / h
            dc_k = (c_k - base_c) / h
            da_k = (a_k - base_a) / h
        dc[k] = dc_k.ravel()
        da[k] = da_k.ravel()
        dD1[k] = (
            (_forward_step(hh.D_ss, base_a + h * da_k, hh.a_grid, hh.pi_s) - D1_base) / h
        ).ravel()
    return dc, da, dD1


def _extract_linear_ssj_paths(
    res: Any,
    *,
    allow_unconverged: bool,
) -> dict[str, Any]:
    """Reconstruct exact first-order Fake News policy/distribution perturbation paths."""
    from puremacro.models.hank_sequence_space import (
        _FD_STEP,
        _egm_step,
        _forward_step,
        _household_block_from_result,
    )

    if not allow_unconverged and not bool(res.ss_converged):
        raise ValueError(
            "SequenceSpaceHANKResult steady state did not converge; pass allow_unconverged=True "
            "to inspect unconverged paths explicitly."
        )

    hh = _household_block_from_result(res)
    dC = np.asarray(res.irf_consumption, dtype=np.float64)
    dY = np.asarray(res.irf_output, dtype=np.float64)
    dr_ea = np.asarray(res.irf_rate, dtype=np.float64)
    T = len(dC)

    fd_h = float(_FD_STEP)
    args = (hh.a_grid, hh.s_grid, hh.pi_s, hh.beta, hh.gamma)
    _, _, base_a = _egm_step(hh.V_ss, *args, hh.r_ss, hh.net_wage(hh.r_ss, hh.w_ss))

    dc_r, da_r, dD1_r = _fake_news_policy_steps(hh, "r", T + 1, h=fd_h)
    dc_w, da_w, dD1_w = _fake_news_policy_steps(hh, "w", T, h=fd_h)
    dr_real = np.concatenate(([0.0], dr_ea))
    dw = dY * (hh.w_ss / hh.C_ss)

    n_a, n_s = hh.c_ss.shape
    dc_path = np.empty((T, n_a, n_s), dtype=np.float64)
    da_path = np.empty((T, n_a, n_s), dtype=np.float64)
    dD_path = np.empty((T, n_a, n_s), dtype=np.float64)
    agg_dA = np.empty(T, dtype=np.float64)
    dD_tot = np.zeros((n_a, n_s), dtype=np.float64)
    for t in range(T):
        rem_r = dr_real[t : T + 1]
        rem_w = dw[t:T]
        dc_flat = rem_r @ dc_r[: len(rem_r)] + rem_w @ dc_w[: len(rem_w)]
        da_flat = rem_r @ da_r[: len(rem_r)] + rem_w @ da_w[: len(rem_w)]
        dD_pol_flat = rem_r @ dD1_r[: len(rem_r)] + rem_w @ dD1_w[: len(rem_w)]
        dc_path[t] = dc_flat.reshape(n_a, n_s)
        da_path[t] = da_flat.reshape(n_a, n_s)
        dD_path[t] = dD_tot
        agg_dA[t] = float(np.sum(hh.D_ss * da_path[t] + dD_tot * hh.a_ss))
        if t + 1 < T:
            dD_tot = _forward_step(dD_tot, base_a, hh.a_grid, hh.pi_s) + dD_pol_flat.reshape(n_a, n_s)

    return {
        "linear": True,
        "model_kind": "sequence_space_linear",
        "closure": "balanced_budget_interest",
        "tax_type": "proportional",
        "validation_scope": "linearized_sequence_space",
        "horizon": T,
        "hh": hh,
        "a_grid": hh.a_grid,
        "s_grid": hh.s_grid,
        "pi_s": hh.pi_s,
        "r_ss": hh.r_ss,
        "c_ss": hh.c_ss,
        "a_ss": hh.a_ss,
        "base_a": base_a,
        "D_ss": hh.D_ss,
        "dc_path": dc_path,
        "da_path": da_path,
        "dD_path": dD_path,
        "da_r": da_r,
        "da_w": da_w,
        "dr_real": dr_real,
        "dw": dw,
        "fd_h": fd_h,
        "agg_dC": dC,
        "agg_dA": agg_dA,
        "agg_dB": None,
        "extra_ss": {
            "Y_ss": hh.C_ss,
            "C_ss": hh.C_ss,
            "G_ss": 0.0,
            "B_ss": hh.B,
            "w_ss": hh.w_ss,
            "beta": hh.beta,
            "gamma": hh.gamma,
            "phi_pi": float(res.phi_pi),
            "kappa": float(res.kappa),
        },
        "source_diag": None,
        "source_converged": bool(res.ss_converged),
        "source_iterations": 1,
        "source_tol": 1e-12,
        "source_method": "sequence_space_linear",
    }


def _extract_model_paths(res: Any, *, allow_unconverged: bool) -> dict[str, Any]:
    """Dispatch extraction across FiscalHANKResult, NonlinearHANKResult, SequenceSpaceHANKResult."""
    from puremacro.models.hank_fiscal import FiscalHANKResult
    from puremacro.models.hank_sequence_space import NonlinearHANKResult, SequenceSpaceHANKResult

    if isinstance(res, FiscalHANKResult):
        return _extract_fiscal_paths(res, allow_unconverged=allow_unconverged)
    if isinstance(res, NonlinearHANKResult):
        return _extract_nonlinear_ssj_paths(res, allow_unconverged=allow_unconverged)
    if isinstance(res, SequenceSpaceHANKResult):
        return _extract_linear_ssj_paths(res, allow_unconverged=allow_unconverged)
    raise TypeError(
        "Unsupported HANK model result type "
        f"{type(res).__name__}; expected FiscalHANKResult, NonlinearHANKResult, "
        "or SequenceSpaceHANKResult."
    )


# ---------------------------------------------------------------------------
# Group trajectory calculation & fail-closed conservation auditing
# ---------------------------------------------------------------------------


def _compute_group_trajectories_nonlinear(
    extracted: dict[str, Any],
    W: np.ndarray,
    masses: np.ndarray,
    mean_c_ss: np.ndarray,
    c_shares_ss: np.ndarray,
    *,
    tracking: str,
    quantile_edges: Sequence[float] | None,
) -> dict[str, np.ndarray]:
    """Compute exact (T, G) group consumption and net-asset IRFs on nonlinear paths."""
    from puremacro.models.hank_sequence_space import _forward_step

    T = int(extracted["horizon"])
    G = W.shape[0]
    a_grid = extracted["a_grid"]
    pi_s = extracted["pi_s"]
    c_ss = extracted["c_ss"]
    a_ss = extracted["a_ss"]
    D_ss = extracted["D_ss"]
    c_path = extracted["c_path"]
    a_path = extracted["a_path"]
    n_s = c_ss.shape[1]

    C_ss_tot = float(np.sum(D_ss * c_ss))
    A_ss_tot = float(np.sum(D_ss * a_ss))
    a_shares_ss = np.einsum("gij,ij->g", W, D_ss * a_ss) / A_ss_tot if abs(A_ss_tot) > 1e-12 else np.zeros(G)

    dc_g = np.empty((T, G), dtype=np.float64)
    dc_pct_g = np.empty((T, G), dtype=np.float64)
    ds_c_g = np.empty((T, G), dtype=np.float64)
    da_g = np.empty((T, G), dtype=np.float64)
    ds_a_g = np.empty((T, G), dtype=np.float64)
    masses_path = np.empty((T, G), dtype=np.float64)

    if tracking == "cohort":
        D_g = np.array([W[g] * D_ss for g in range(G)], dtype=np.float64)
        D_g_ss = np.array([W[g] * D_ss for g in range(G)], dtype=np.float64)
        for t in range(T):
            D_tot_t = np.sum(D_g, axis=0)
            C_t = float(np.sum(D_tot_t * c_path[t]))
            A_t = float(np.sum(D_tot_t * a_path[t]))
            c_tot_g = np.einsum("gij,ij->g", D_g, c_path[t])
            c_tot_base = np.einsum("gij,ij->g", D_g_ss, c_ss)
            a_tot_g = np.einsum("gij,ij->g", D_g, a_path[t])
            a_tot_base = np.einsum("gij,ij->g", D_g_ss, a_ss)

            masses_path[t] = np.sum(D_g, axis=(1, 2))
            dc_g[t] = (c_tot_g - c_tot_base) / masses
            dc_pct_g[t] = dc_g[t] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
            ds_c_g[t] = (c_tot_g / C_t) - (c_tot_base / C_ss_tot)
            da_g[t] = (a_tot_g - a_tot_base) / masses
            if abs(A_t) > 1e-12 and abs(A_ss_tot) > 1e-12:
                ds_a_g[t] = (a_tot_g / A_t) - (a_tot_base / A_ss_tot)
            else:
                ds_a_g[t] = 0.0

            if t + 1 < T:
                for g in range(G):
                    D_g[g] = _forward_step(D_g[g], a_path[t], a_grid, pi_s)
                    D_g_ss[g] = _forward_step(D_g_ss[g], a_ss, a_grid, pi_s)
    elif tracking in ("cross_section", "fixed_bins"):
        if tracking == "cross_section" and quantile_edges is None:
            raise ValueError(
                "tracking='cross_section' is only defined for quantile-based wealth partitions "
                "(e.g. 'quintiles', 'deciles', 'quartiles', integer G, or quantile edges); "
                "use tracking='cohort' or 'fixed_bins' for state-dependent groups."
            )
        D_t = D_ss.copy()
        c_tot_ss = np.einsum("gij,ij->g", W, D_ss * c_ss)
        a_tot_ss = np.einsum("gij,ij->g", W, D_ss * a_ss)
        for t in range(T):
            if tracking == "cross_section":
                assert quantile_edges is not None
                W_a_t = _quantile_edges_weights(D_t.sum(axis=1), quantile_edges)
                W_t = np.repeat(W_a_t[:, :, None], n_s, axis=2)
            else:
                W_t = W

            C_t = float(np.sum(D_t * c_path[t]))
            A_t = float(np.sum(D_t * a_path[t]))
            c_tot_g = np.einsum("gij,ij->g", W_t, D_t * c_path[t])
            a_tot_g = np.einsum("gij,ij->g", W_t, D_t * a_path[t])

            masses_path[t] = np.einsum("gij,ij->g", W_t, D_t)
            dc_g[t] = (c_tot_g - c_tot_ss) / masses
            dc_pct_g[t] = dc_g[t] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
            ds_c_g[t] = (c_tot_g / C_t) - c_shares_ss
            da_g[t] = (a_tot_g - a_tot_ss) / masses
            if abs(A_t) > 1e-12 and abs(A_ss_tot) > 1e-12:
                ds_a_g[t] = (a_tot_g / A_t) - a_shares_ss
            else:
                ds_a_g[t] = 0.0

            if t + 1 < T:
                D_t = _forward_step(D_t, a_path[t], a_grid, pi_s)
    else:
        raise ValueError(
            f"Unknown tracking {tracking!r}; expected 'cohort', 'cross_section', or 'fixed_bins'."
        )

    return {
        "consumption_irf": dc_g,
        "consumption_pct_irf": dc_pct_g,
        "consumption_share_irf": ds_c_g,
        "asset_irf": da_g,
        "asset_share_irf": ds_a_g,
        "group_masses_path": masses_path,
    }


def _compute_group_trajectories_linear(
    extracted: dict[str, Any],
    W: np.ndarray,
    masses: np.ndarray,
    mean_c_ss: np.ndarray,
    c_shares_ss: np.ndarray,
    *,
    tracking: str,
) -> dict[str, np.ndarray]:
    """Compute exact first-order (T, G) group IRFs from Fake News policy/distribution steps."""
    from puremacro.models.hank_sequence_space import _forward_step, _lottery

    if tracking == "cross_section":
        raise ValueError(
            "tracking='cross_section' requires a nonlinear transition result "
            "(FiscalHANKResult or NonlinearHANKResult); use tracking='cohort' or 'fixed_bins' "
            "for linearized SequenceSpaceHANKResult."
        )
    if tracking not in ("cohort", "fixed_bins"):
        raise ValueError(
            f"Unknown tracking {tracking!r}; expected 'cohort', 'cross_section', or 'fixed_bins'."
        )

    T = int(extracted["horizon"])
    G = W.shape[0]
    a_grid = extracted["a_grid"]
    pi_s = extracted["pi_s"]
    c_ss = extracted["c_ss"]
    a_ss = extracted["a_ss"]
    base_a = extracted["base_a"]
    D_ss = extracted["D_ss"]
    dc_path = extracted["dc_path"]
    da_path = extracted["da_path"]
    dD_path = extracted["dD_path"]
    agg_dC = extracted["agg_dC"]
    agg_dA = extracted["agg_dA"]
    n_a, n_s = c_ss.shape

    C_ss_tot = float(np.sum(D_ss * c_ss))
    A_ss_tot = float(np.sum(D_ss * a_ss))
    a_shares_ss = np.einsum("gij,ij->g", W, D_ss * a_ss) / A_ss_tot if abs(A_ss_tot) > 1e-12 else np.zeros(G)

    dc_g = np.empty((T, G), dtype=np.float64)
    dc_pct_g = np.empty((T, G), dtype=np.float64)
    ds_c_g = np.empty((T, G), dtype=np.float64)
    da_g = np.empty((T, G), dtype=np.float64)
    ds_a_g = np.empty((T, G), dtype=np.float64)
    masses_path = np.empty((T, G), dtype=np.float64)

    if tracking == "cohort":
        fd_h = float(extracted["fd_h"])
        da_r = extracted["da_r"]
        da_w = extracted["da_w"]
        dr_real = extracted["dr_real"]
        dw = extracted["dw"]
        l0 = _lottery(base_a, a_grid)
        lr = [_lottery(base_a + fd_h * da_r[k].reshape(n_a, n_s), a_grid) for k in range(T + 1)]
        lw = [_lottery(base_a + fd_h * da_w[k].reshape(n_a, n_s), a_grid) for k in range(T)]
        rows_a = np.arange(n_a)

        D_g_ss = np.array([W[g] * D_ss for g in range(G)], dtype=np.float64)
        dD_g = np.zeros_like(D_g_ss)
        for t in range(T):
            dC_tot_g = np.einsum("gij,ij->g", D_g_ss, dc_path[t]) + np.einsum("gij,ij->g", dD_g, c_ss)
            dA_tot_g = np.einsum("gij,ij->g", D_g_ss, da_path[t]) + np.einsum("gij,ij->g", dD_g, a_ss)
            c_tot_base = np.einsum("gij,ij->g", D_g_ss, c_ss)
            a_tot_base = np.einsum("gij,ij->g", D_g_ss, a_ss)
            s_c_base = c_tot_base / C_ss_tot

            masses_path[t] = np.sum(D_g_ss, axis=(1, 2)) + np.sum(dD_g, axis=(1, 2))
            dc_g[t] = dC_tot_g / masses
            dc_pct_g[t] = dc_g[t] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
            ds_c_g[t] = (dC_tot_g - s_c_base * agg_dC[t]) / C_ss_tot
            da_g[t] = dA_tot_g / masses
            if abs(A_ss_tot) > 1e-12:
                s_a_base = a_tot_base / A_ss_tot
                ds_a_g[t] = (dA_tot_g - s_a_base * agg_dA[t]) / A_ss_tot
            else:
                ds_a_g[t] = 0.0

            if t + 1 < T:
                rem_r = dr_real[t : T + 1]
                rem_w = dw[t:T]
                dD_pol_g = np.zeros_like(D_g_ss)
                c_sum = (float(np.sum(rem_r)) + float(np.sum(rem_w))) / fd_h
                for j in range(n_s):
                    S_j = np.zeros((n_a, n_a), dtype=np.float64)
                    S_j[l0[0][:, j], rows_a] -= c_sum * l0[2][:, j]
                    S_j[l0[1][:, j], rows_a] -= c_sum * l0[3][:, j]
                    for k in range(len(rem_r)):
                        ck = float(rem_r[k]) / fd_h
                        if ck != 0.0:
                            il, ih, wl, wh = lr[k]
                            np.add.at(S_j, (il[:, j], rows_a), ck * wl[:, j])
                            np.add.at(S_j, (ih[:, j], rows_a), ck * wh[:, j])
                    for k in range(len(rem_w)):
                        ck = float(rem_w[k]) / fd_h
                        if ck != 0.0:
                            il, ih, wl, wh = lw[k]
                            np.add.at(S_j, (il[:, j], rows_a), ck * wl[:, j])
                            np.add.at(S_j, (ih[:, j], rows_a), ck * wh[:, j])
                    dD_pol_g[:, :, j] = D_g_ss[:, :, j] @ S_j.T
                dD_pol_g = dD_pol_g @ pi_s
                for g in range(G):
                    dD_g[g] = _forward_step(dD_g[g], base_a, a_grid, pi_s) + dD_pol_g[g]
                    D_g_ss[g] = _forward_step(D_g_ss[g], base_a, a_grid, pi_s)
    else:
        for t in range(T):
            dD = dD_path[t]
            dC_tot_g = np.einsum("gij,ij->g", W, D_ss * dc_path[t] + dD * c_ss)
            dA_tot_g = np.einsum("gij,ij->g", W, D_ss * da_path[t] + dD * a_ss)
            masses_path[t] = np.einsum("gij,ij->g", W, D_ss + dD)
            dc_g[t] = dC_tot_g / masses
            dc_pct_g[t] = dc_g[t] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
            ds_c_g[t] = (dC_tot_g - c_shares_ss * agg_dC[t]) / C_ss_tot
            da_g[t] = dA_tot_g / masses
            if abs(A_ss_tot) > 1e-12:
                ds_a_g[t] = (dA_tot_g - a_shares_ss * agg_dA[t]) / A_ss_tot
            else:
                ds_a_g[t] = 0.0

    return {
        "consumption_irf": dc_g,
        "consumption_pct_irf": dc_pct_g,
        "consumption_share_irf": ds_c_g,
        "asset_irf": da_g,
        "asset_share_irf": ds_a_g,
        "group_masses_path": masses_path,
    }


def _audit_conservation_and_build_diagnostics(
    extracted: dict[str, Any],
    trajectories: dict[str, np.ndarray],
    masses: np.ndarray,
    calibration: dict[str, Any],
    *,
    conservation_tol: float,
    strict_conservation: bool,
) -> tuple[dict[str, float], SolverDiagnosticRecord]:
    """Audit all cross-sectional conservation identities and construct SolverDiagnosticRecord."""
    dc_g = trajectories["consumption_irf"]
    ds_c_g = trajectories["consumption_share_irf"]
    da_g = trajectories["asset_irf"]
    ds_a_g = trajectories["asset_share_irf"]
    masses_path = trajectories["group_masses_path"]
    agg_dC = extracted["agg_dC"]
    agg_dA = extracted["agg_dA"]
    agg_dB = extracted.get("agg_dB")

    all_arrays = (dc_g, ds_c_g, da_g, ds_a_g, masses_path, agg_dC, agg_dA)
    if any(not np.all(np.isfinite(arr)) for arr in all_arrays):
        raise ValueError("Non-finite values encountered in HANK group IRF trajectories")

    cons_err = float(np.max(np.abs(dc_g @ masses - agg_dC)))
    c_share_err = float(np.max(np.abs(np.sum(ds_c_g, axis=1))))
    asset_err = float(np.max(np.abs(da_g @ masses - agg_dA)))
    a_share_err = float(np.max(np.abs(np.sum(ds_a_g, axis=1))))
    mass_err = float(np.max(np.abs(np.sum(masses_path, axis=1) - 1.0)))
    asset_vs_debt_err = (
        float(np.max(np.abs(da_g @ masses - np.asarray(agg_dB, dtype=np.float64))))
        if agg_dB is not None
        else float("nan")
    )

    conservation_audit: dict[str, float] = {
        "consumption_adding_up_error": cons_err,
        "consumption_share_adding_up_error": c_share_err,
        "asset_adding_up_error": asset_err,
        "asset_share_adding_up_error": a_share_err,
        "population_mass_error": mass_err,
        "asset_vs_debt_error": asset_vs_debt_err,
        "tolerance": float(conservation_tol),
    }

    max_cons_violation = max(cons_err, c_share_err, asset_err, a_share_err, mass_err)
    conservation_ok = max_cons_violation < float(conservation_tol)

    if strict_conservation and not conservation_ok:
        raise ValueError(
            "Cross-sectional accounting conservation failed "
            f"(tol={conservation_tol:.1e}): "
            f"dC_err={cons_err:.3e}, dShareC_err={c_share_err:.3e}, "
            f"dA_err={asset_err:.3e}, dShareA_err={a_share_err:.3e}, mass_err={mass_err:.3e}."
        )

    source_diag: SolverDiagnosticRecord | None = extracted.get("source_diag")
    if source_diag is not None:
        conv_ok = bool(source_diag.convergence.converged)
        iters = int(source_diag.convergence.iterations)
        stop_reason = str(source_diag.convergence.stopping_reason)
        method = str(source_diag.convergence.solver_method)
        conv_tol = float(source_diag.convergence.tolerance)
        src_acc_ok = bool(source_diag.accuracy.is_accurate)
        src_max_res = float(source_diag.accuracy.max_residual)
        src_residuals = dict(source_diag.accuracy.residuals)
        src_supp_ok = bool(source_diag.support.is_supported)
        regime = str(source_diag.support.regime)
        supp_warnings = list(source_diag.support.warnings)
        boundary_metrics = dict(source_diag.support.boundary_metrics)
    else:
        conv_ok = bool(extracted["source_converged"])
        iters = int(extracted["source_iterations"])
        stop_reason = "tolerance_met" if conv_ok else "not_converged"
        method = str(extracted["source_method"])
        conv_tol = float(extracted["source_tol"])
        src_max_res = float(extracted["extra_ss"].get("goods_market_residual", 0.0))
        src_acc_ok = conv_ok and src_max_res <= max(conv_tol, float(conservation_tol))
        src_residuals = {"goods_market_inf": src_max_res} if src_max_res > 0.0 else {}
        top_mass = float(calibration["top_grid_mass"])
        src_supp_ok = top_mass <= _TOP_GRID_MASS_WARN
        regime = "interior" if src_supp_ok else "boundary_asset_truncation"
        supp_warnings = (
            [
                f"Upper asset grid truncation detected (top_grid_mass={top_mass:.2e} > {_TOP_GRID_MASS_WARN:.1e}); "
                "increase a_max >= 100.0."
            ]
            if not src_supp_ok
            else []
        )
        boundary_metrics = {"top_grid_mass": top_mass}

    residuals = {
        **src_residuals,
        "group_consumption_adding_up": cons_err,
        "group_consumption_share_adding_up": c_share_err,
        "group_asset_adding_up": asset_err,
        "group_asset_share_adding_up": a_share_err,
        "group_population_mass": mass_err,
    }
    if math.isfinite(asset_vs_debt_err):
        residuals["group_asset_vs_debt"] = asset_vs_debt_err

    is_accurate = bool(src_acc_ok and conservation_ok)
    max_res_all = max(src_max_res, max_cons_violation)

    diag = make_solver_diagnostic_record(
        converged=conv_ok,
        iterations=iters,
        stopping_reason=stop_reason,
        solver_method=f"hank_distributional_bridge::{method}",
        tolerance=conv_tol,
        is_accurate=is_accurate,
        max_residual=float(max_res_all),
        residual_norm=float(np.linalg.norm([cons_err, c_share_err, asset_err, mass_err])),
        residual_type="cross_sectional_conservation_and_market_clearing",
        accuracy_tolerance=float(conservation_tol),
        residuals=residuals,
        is_supported=src_supp_ok,
        regime=regime,
        warnings=supp_warnings,
        boundary_metrics=boundary_metrics,
        evidence={
            "validation_scope": extracted["validation_scope"],
            "model_kind": extracted["model_kind"],
            "closure": extracted["closure"],
            "tax_type": extracted["tax_type"],
        },
    )
    return conservation_audit, diag


# ---------------------------------------------------------------------------
# Public extraction & simulation entry points
# ---------------------------------------------------------------------------


def extract_hank_group_irf(
    model_result: Any,
    *,
    groups: str | int | Sequence[float] | Mapping[str, np.ndarray] = "quintiles",
    group_labels: Sequence[str] | None = None,
    htm_definition: Literal["constrained", "bottom_quartile"] = "constrained",
    tracking: Literal["cohort", "cross_section", "fixed_bins"] = "cohort",
    conservation_tol: float = _DEFAULT_CONSERVATION_TOL,
    strict_conservation: bool = True,
    allow_unconverged: bool = False,
) -> HANKGroupIRFResult:
    """Extract conservation-verified household-group IRFs from a solved HANK result.

    Parameters
    ----------
    model_result : FiscalHANKResult | NonlinearHANKResult | SequenceSpaceHANKResult
        Converged HANK equilibrium transition result.
    groups : str | int | Sequence[float] | Mapping[str, np.ndarray], default 'quintiles'
        Cross-sectional household partition: ``'quintiles'``, ``'deciles'``,
        ``'quartiles'``, ``'terciles'``, ``'htm_vs_unconstrained'``, ``'income'``,
        ``'wealth_x_income'``, integer bin count ``G >= 2``, cumulative-mass
        quantile edges, or mapping of group names to grid weights.
    group_labels : Sequence[str] or None, optional
        Optional custom labels matching the ``G`` groups.
    htm_definition : {'constrained', 'bottom_quartile'}, default 'constrained'
        Definition used when ``groups='htm'`` or ``'htm_vs_unconstrained'``.
    tracking : {'cohort', 'cross_section', 'fixed_bins'}, default 'cohort'
        Dynamic tracking convention across horizons ``t = 0, ..., T - 1``.
    conservation_tol : float, default 1e-10
        Maximum allowed error on population-weighted adding-up and share zero-sum.
    strict_conservation : bool, default True
        When ``True``, raises ``ValueError`` if any conservation check exceeds
        ``conservation_tol`` (fail closed, no silent rescaling).
    allow_unconverged : bool, default False
        When ``False``, refuses unconverged HANK results with ``ValueError``.
    """
    tol_val = float(conservation_tol)
    if not (math.isfinite(tol_val) and tol_val > 0.0):
        raise ValueError(f"conservation_tol must be a positive finite float, got {conservation_tol!r}")

    extracted = _extract_model_paths(model_result, allow_unconverged=allow_unconverged)
    names, W, scheme, quantile_edges = _build_group_partition(
        extracted["D_ss"],
        extracted["a_ss"],
        extracted["a_grid"],
        groups=groups,
        group_labels=group_labels,
        htm_definition=htm_definition,
    )
    cal, masses, mean_c_ss, mean_a_ss, group_mpc, c_shares_ss = _compute_steady_state_calibration(
        extracted["D_ss"],
        extracted["c_ss"],
        extracted["a_ss"],
        extracted["a_grid"],
        extracted["r_ss"],
        W,
        names,
        extracted["extra_ss"],
    )

    if extracted["linear"]:
        trajectories = _compute_group_trajectories_linear(
            extracted,
            W,
            masses,
            mean_c_ss,
            c_shares_ss,
            tracking=tracking,
        )
    else:
        trajectories = _compute_group_trajectories_nonlinear(
            extracted,
            W,
            masses,
            mean_c_ss,
            c_shares_ss,
            tracking=tracking,
            quantile_edges=quantile_edges,
        )

    conservation_audit, diag = _audit_conservation_and_build_diagnostics(
        extracted,
        trajectories,
        masses,
        cal,
        conservation_tol=tol_val,
        strict_conservation=strict_conservation,
    )

    return HANKGroupIRFResult(
        horizons=np.arange(int(extracted["horizon"]), dtype=np.int64),
        groups=names,
        group_scheme=scheme,
        tracking=str(tracking),
        model_kind=str(extracted["model_kind"]),
        closure=str(extracted["closure"]),
        tax_type=str(extracted["tax_type"]),
        consumption_irf=trajectories["consumption_irf"],
        consumption_pct_irf=trajectories["consumption_pct_irf"],
        consumption_share_irf=trajectories["consumption_share_irf"],
        asset_irf=trajectories["asset_irf"],
        asset_share_irf=trajectories["asset_share_irf"],
        aggregate_consumption_irf=extracted["agg_dC"],
        aggregate_asset_irf=extracted["agg_dA"],
        group_masses=masses,
        group_masses_path=trajectories["group_masses_path"],
        steady_state_group_consumption=mean_c_ss,
        steady_state_group_assets=mean_a_ss,
        steady_state_group_mpc=group_mpc,
        steady_state_group_consumption_shares=c_shares_ss,
        conservation_audit=conservation_audit,
        calibration=cal,
        solver_diagnostics=diag,
        validation_scope=str(extracted["validation_scope"]),
        source_result=model_result,
    )


def solve_hank_group_irf(
    *,
    ss_model: Any | None = None,
    shock_seq: Sequence[float] | np.ndarray | None = None,
    monetary_shock_seq: Sequence[float] | np.ndarray | None = None,
    closure: str = "balanced_budget",
    tax_type: str = "proportional",
    horizon: int = 80,
    phi_B: float = 0.08,
    phi_G: float = 0.0,
    g_ss: float = 0.0,
    beta: float = 0.985,
    gamma: float = 1.0,
    r_ss: float = 0.01,
    phi_pi: float = 1.5,
    kappa: float = 0.1,
    n_a: int = 100,
    a_max: float = 100.0,
    tol: float = 1e-9,
    linear: bool = False,
    groups: str | int | Sequence[float] | Mapping[str, np.ndarray] = "quintiles",
    group_labels: Sequence[str] | None = None,
    htm_definition: Literal["constrained", "bottom_quartile"] = "constrained",
    tracking: Literal["cohort", "cross_section", "fixed_bins"] = "cohort",
    conservation_tol: float = _DEFAULT_CONSERVATION_TOL,
    strict_conservation: bool = True,
    allow_unconverged: bool = False,
) -> HANKGroupIRFResult:
    """Solve one-asset HANK transition and extract conservation-audited household-group IRFs.

    Defaults to ``n_a=100, a_max=100.0`` to avoid upper-tail asset truncation.
    When ``linear=False`` (default), uses :func:`puremacro.models.hank_fiscal.solve_hank_fiscal`
    for financed fiscal closures (``balanced_budget``, ``debt_rule``) and/or monetary
    shocks (pass ``shock_seq=np.zeros(horizon), monetary_shock_seq=...`` for a pure
    monetary shock under an explicit fiscal closure). When ``linear=True``, solves
    :func:`puremacro.models.hank_sequence_space.solve_hank_sequence_space` (monetary
    shock only; ``shock_seq`` must be ``None``).
    """
    if linear:
        from puremacro.models.hank_sequence_space import solve_hank_sequence_space

        if shock_seq is not None:
            raise ValueError(
                "linear=True solves the linear sequence-space monetary equilibrium; "
                "use linear=False for fiscal spending/tax closures with shock_seq."
            )
        m_arr = _as_finite_array(monetary_shock_seq, "monetary_shock_seq", ndim=1) if monetary_shock_seq is not None else None
        if m_arr is not None:
            if len(m_arr) >= 2 and abs(float(m_arr[0])) > 1e-15:
                rho_est = float(m_arr[1] / m_arr[0])
            else:
                rho_est = 0.0
            mag = float(m_arr[0])
            expected = mag * (rho_est ** np.arange(len(m_arr)))
            if len(m_arr) > 1 and float(np.max(np.abs(m_arr - expected))) > 1e-10:
                raise ValueError(
                    "solve_hank_sequence_space accepts AR(1) monetary shocks; pass an AR(1) "
                    "monetary_shock_seq or set linear=False for arbitrary shock sequences."
                )
            ss_lin = solve_hank_sequence_space(
                T=max(int(horizon), len(m_arr)),
                beta=beta,
                gamma=gamma,
                r_ss=r_ss,
                phi_pi=phi_pi,
                kappa=kappa,
                shock_magnitude=mag,
                shock_rho=rho_est,
                n_a=n_a,
                a_max=a_max,
            )
        else:
            ss_lin = ss_model if ss_model is not None else solve_hank_sequence_space(
                T=int(horizon),
                beta=beta,
                gamma=gamma,
                r_ss=r_ss,
                phi_pi=phi_pi,
                kappa=kappa,
                n_a=n_a,
                a_max=a_max,
            )
        return extract_hank_group_irf(
            ss_lin,
            groups=groups,
            group_labels=group_labels,
            htm_definition=htm_definition,
            tracking=tracking,
            conservation_tol=conservation_tol,
            strict_conservation=strict_conservation,
            allow_unconverged=allow_unconverged,
        )

    from puremacro.models.hank_fiscal import solve_hank_fiscal

    res = solve_hank_fiscal(
        ss_model=ss_model,
        shock_seq=shock_seq,
        closure=closure,
        tax_type=tax_type,
        horizon=horizon,
        phi_B=phi_B,
        phi_G=phi_G,
        g_ss=g_ss,
        monetary_shock_seq=monetary_shock_seq,
        beta=beta,
        gamma=gamma,
        r_ss=r_ss,
        phi_pi=phi_pi,
        kappa=kappa,
        n_a=n_a,
        a_max=a_max,
        tol=tol,
    )
    return extract_hank_group_irf(
        res,
        groups=groups,
        group_labels=group_labels,
        htm_definition=htm_definition,
        tracking=tracking,
        conservation_tol=conservation_tol,
        strict_conservation=strict_conservation,
        allow_unconverged=allow_unconverged,
    )


def simulate_transfer_group_irf(
    *,
    ss_model: Any | None = None,
    target: str | Sequence[int] = "all",
    amount: float = 0.01,
    horizon: int = 40,
    nonlinear: bool = False,
    groups: str | int | Sequence[float] | Mapping[str, np.ndarray] = "quintiles",
    group_labels: Sequence[str] | None = None,
    htm_definition: Literal["constrained", "bottom_quartile"] = "constrained",
    tracking: Literal["cohort", "cross_section", "fixed_bins"] = "cohort",
    beta: float = 0.985,
    gamma: float = 1.0,
    r_ss: float = 0.01,
    n_a: int = 100,
    a_max: float = 100.0,
    conservation_tol: float = _DEFAULT_CONSERVATION_TOL,
    strict_conservation: bool = True,
) -> HANKGroupIRFResult:
    """Simulate a partial-equilibrium date-0 cash-on-hand transfer across household groups.

    Mirrors :func:`puremacro.models.hank_sequence_space.simulate_targeted_transfer`
    while propagating full ``(T, G)`` group consumption and net-asset impulse responses
    with exact adding-up conservation.
    """
    from puremacro.models.hank_sequence_space import (
        SequenceSpaceHANKResult,
        _decile_weights,
        _forward_step,
        _income_process,
        _interp_extrap,
        _local_mpc,
        _mass_window_weights,
        solve_hank_sequence_space,
    )

    T = int(horizon)
    if T < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon!r}")
    amt = float(amount)
    if not (math.isfinite(amt) and amt > 0.0):
        raise ValueError(f"amount must be a positive finite float, got {amount!r}")
    tol_val = float(conservation_tol)
    if not (math.isfinite(tol_val) and tol_val > 0.0):
        raise ValueError(f"conservation_tol must be positive and finite, got {conservation_tol!r}")

    if ss_model is None:
        ss = solve_hank_sequence_space(
            T=T,
            beta=beta,
            gamma=gamma,
            r_ss=r_ss,
            n_a=n_a,
            a_max=a_max,
        )
    elif isinstance(ss_model, SequenceSpaceHANKResult):
        ss = ss_model
    else:
        raise TypeError(f"ss_model must be SequenceSpaceHANKResult or None, got {type(ss_model).__name__}")

    D_ss = np.asarray(ss.distribution, dtype=np.float64)
    c_ss = np.asarray(ss.policy_c, dtype=np.float64)
    a_ss = np.asarray(ss.policy_a, dtype=np.float64)
    a_grid = np.asarray(ss.asset_grid, dtype=np.float64)
    r_ss_val = float(ss.r_ss)
    n_a_val, n_s = c_ss.shape
    _, pi_s = _income_process(n_s)
    D_a = D_ss.sum(axis=1)
    W_dec = _decile_weights(D_a, n_bins=10)

    if isinstance(target, str):
        tgt = target.lower().strip()
        if tgt in ("borrowers", "hand_to_mouth", "constrained", "bottom_quartile", "p25"):
            elig = _mass_window_weights(D_a, 0.0, 0.25)
        elif tgt in ("unconstrained", "wealthy"):
            elig = _mass_window_weights(D_a, 0.5, 1.0)
        elif tgt in ("all", "universal", "lump_sum"):
            elig = np.ones(n_a_val, dtype=np.float64)
        else:
            raise ValueError(
                f"Unknown target {target!r}; choose 'borrowers', 'unconstrained', 'all', "
                "or decile indices 1..10."
            )
    else:
        deciles = [int(d) for d in target]
        if not deciles or any(d < 1 or d > 10 for d in deciles):
            raise ValueError("target deciles must be integers in 1..10")
        elig = np.clip(W_dec[[d - 1 for d in sorted(set(deciles))]].sum(axis=0), 0.0, 1.0)

    eligible_mass = float(np.sum(elig * D_a))
    if eligible_mass <= 0.0:
        raise ValueError(f"target {target!r} has zero mass")
    tau = amt / eligible_mass
    transfer_grid = np.repeat((elig * tau)[:, None], n_s, axis=1)

    names, W, scheme, quantile_edges = _build_group_partition(
        D_ss,
        a_ss,
        a_grid,
        groups=groups,
        group_labels=group_labels,
        htm_definition=htm_definition,
    )
    cal, masses, mean_c_ss, mean_a_ss, group_mpc, c_shares_ss = _compute_steady_state_calibration(
        D_ss,
        c_ss,
        a_ss,
        a_grid,
        r_ss_val,
        W,
        names,
        {
            "Y_ss": float(ss.steady_state_consumption),
            "C_ss": float(ss.steady_state_consumption),
            "G_ss": 0.0,
            "B_ss": float(ss.government_debt),
            "w_ss": float(ss.w_ss),
            "beta": float(ss.beta),
            "gamma": float(ss.gamma),
            "transfer_amount": amt,
            "transfer_target": str(target),
        },
    )

    G = W.shape[0]
    C_ss_tot = float(np.sum(D_ss * c_ss))
    A_ss_tot = float(np.sum(D_ss * a_ss))
    a_shares_ss = np.einsum("gij,ij->g", W, D_ss * a_ss) / A_ss_tot if abs(A_ss_tot) > 1e-12 else np.zeros(G)

    dc_g = np.empty((T, G), dtype=np.float64)
    dc_pct_g = np.empty((T, G), dtype=np.float64)
    ds_c_g = np.empty((T, G), dtype=np.float64)
    da_g = np.empty((T, G), dtype=np.float64)
    ds_a_g = np.empty((T, G), dtype=np.float64)
    masses_path = np.empty((T, G), dtype=np.float64)
    agg_dC = np.empty(T, dtype=np.float64)
    agg_dA = np.empty(T, dtype=np.float64)

    if not nonlinear:
        if tracking == "cross_section":
            raise ValueError(
                "tracking='cross_section' requires nonlinear=True in simulate_transfer_group_irf."
            )
        mpc = _local_mpc(c_ss, a_grid, r_ss_val)
        dc0 = transfer_grid * mpc
        da0 = transfer_grid * (1.0 - mpc)
        direction = elig[:, None] * (1.0 - mpc)
        fd_h = 1e-4

        agg_dC[0] = float(np.sum(D_ss * dc0))
        agg_dA[0] = float(np.sum(D_ss * da0))

        dC0_g = np.einsum("gij,ij->g", W, D_ss * dc0)
        dA0_g = np.einsum("gij,ij->g", W, D_ss * da0)
        masses_path[0] = masses
        dc_g[0] = dC0_g / masses
        dc_pct_g[0] = dc_g[0] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
        ds_c_g[0] = (dC0_g - c_shares_ss * agg_dC[0]) / C_ss_tot
        da_g[0] = dA0_g / masses
        ds_a_g[0] = (dA0_g - a_shares_ss * agg_dA[0]) / A_ss_tot if abs(A_ss_tot) > 1e-12 else 0.0

        if tracking == "cohort":
            D_g_ss = np.array([W[g] * D_ss for g in range(G)], dtype=np.float64)
            dD_g = np.array(
                [
                    tau
                    * (
                        _forward_step(D_g_ss[g], a_ss + fd_h * direction, a_grid, pi_s)
                        - _forward_step(D_g_ss[g], a_ss, a_grid, pi_s)
                    )
                    / fd_h
                    for g in range(G)
                ],
                dtype=np.float64,
            )
            for t in range(1, T):
                D_g_ss = np.array([_forward_step(D_g_ss[g], a_ss, a_grid, pi_s) for g in range(G)])
                dD_tot = np.sum(dD_g, axis=0)
                agg_dC[t] = float(np.sum(dD_tot * c_ss))
                agg_dA[t] = float(np.sum(dD_tot * a_ss))

                dC_t_g = np.einsum("gij,ij->g", dD_g, c_ss)
                dA_t_g = np.einsum("gij,ij->g", dD_g, a_ss)
                s_c_base = np.einsum("gij,ij->g", D_g_ss, c_ss) / C_ss_tot

                masses_path[t] = np.sum(D_g_ss, axis=(1, 2)) + np.sum(dD_g, axis=(1, 2))
                dc_g[t] = dC_t_g / masses
                dc_pct_g[t] = dc_g[t] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
                ds_c_g[t] = (dC_t_g - s_c_base * agg_dC[t]) / C_ss_tot
                da_g[t] = dA_t_g / masses
                if abs(A_ss_tot) > 1e-12:
                    s_a_base = np.einsum("gij,ij->g", D_g_ss, a_ss) / A_ss_tot
                    ds_a_g[t] = (dA_t_g - s_a_base * agg_dA[t]) / A_ss_tot
                else:
                    ds_a_g[t] = 0.0

                if t + 1 < T:
                    dD_g = np.array([_forward_step(dD_g[g], a_ss, a_grid, pi_s) for g in range(G)])
        elif tracking == "fixed_bins":
            dD = (
                tau
                * (
                    _forward_step(D_ss, a_ss + fd_h * direction, a_grid, pi_s)
                    - _forward_step(D_ss, a_ss, a_grid, pi_s)
                )
                / fd_h
            )
            for t in range(1, T):
                agg_dC[t] = float(np.sum(dD * c_ss))
                agg_dA[t] = float(np.sum(dD * a_ss))
                dC_t_g = np.einsum("gij,ij->g", W, dD * c_ss)
                dA_t_g = np.einsum("gij,ij->g", W, dD * a_ss)
                masses_path[t] = np.einsum("gij,ij->g", W, D_ss + dD)
                dc_g[t] = dC_t_g / masses
                dc_pct_g[t] = dc_g[t] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
                ds_c_g[t] = (dC_t_g - c_shares_ss * agg_dC[t]) / C_ss_tot
                da_g[t] = dA_t_g / masses
                ds_a_g[t] = (dA_t_g - a_shares_ss * agg_dA[t]) / A_ss_tot if abs(A_ss_tot) > 1e-12 else 0.0
                if t + 1 < T:
                    dD = _forward_step(dD, a_ss, a_grid, pi_s)
        else:
            raise ValueError(f"Unknown tracking {tracking!r}; expected 'cohort', 'cross_section', or 'fixed_bins'.")
    else:
        shift = tau / (1.0 + r_ss_val)
        c_tr = np.column_stack([_interp_extrap(a_grid + shift, a_grid, c_ss[:, j]) for j in range(n_s)])
        a_tr = np.maximum(
            np.column_stack([_interp_extrap(a_grid + shift, a_grid, a_ss[:, j]) for j in range(n_s)]),
            0.0,
        )
        c0_eff = elig[:, None] * c_tr + (1.0 - elig)[:, None] * c_ss
        a0_eff = elig[:, None] * a_tr + (1.0 - elig)[:, None] * a_ss

        C_0 = float(np.sum(D_ss * c0_eff))
        A_0 = float(np.sum(D_ss * a0_eff))
        agg_dC[0] = C_0 - C_ss_tot
        agg_dA[0] = A_0 - A_ss_tot

        c_tot_0 = np.einsum("gij,ij->g", W, D_ss * c0_eff)
        c_tot_base0 = np.einsum("gij,ij->g", W, D_ss * c_ss)
        a_tot_0 = np.einsum("gij,ij->g", W, D_ss * a0_eff)
        a_tot_base0 = np.einsum("gij,ij->g", W, D_ss * a_ss)
        masses_path[0] = masses
        dc_g[0] = (c_tot_0 - c_tot_base0) / masses
        dc_pct_g[0] = dc_g[0] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
        ds_c_g[0] = c_tot_0 / C_0 - c_shares_ss
        da_g[0] = (a_tot_0 - a_tot_base0) / masses
        ds_a_g[0] = (a_tot_0 / A_0 - a_shares_ss) if (abs(A_0) > 1e-12 and abs(A_ss_tot) > 1e-12) else 0.0

        if tracking == "cohort":
            D_g_ss = np.array([W[g] * D_ss for g in range(G)], dtype=np.float64)
            D_g = np.array(
                [
                    _forward_step(elig[:, None] * D_g_ss[g], a_tr, a_grid, pi_s)
                    + _forward_step((1.0 - elig)[:, None] * D_g_ss[g], a_ss, a_grid, pi_s)
                    for g in range(G)
                ],
                dtype=np.float64,
            )
            for t in range(1, T):
                D_g_ss = np.array([_forward_step(D_g_ss[g], a_ss, a_grid, pi_s) for g in range(G)])
                D_tot_t = np.sum(D_g, axis=0)
                C_t = float(np.sum(D_tot_t * c_ss))
                A_t = float(np.sum(D_tot_t * a_ss))
                agg_dC[t] = C_t - C_ss_tot
                agg_dA[t] = A_t - A_ss_tot

                c_tot_g = np.einsum("gij,ij->g", D_g, c_ss)
                c_tot_base = np.einsum("gij,ij->g", D_g_ss, c_ss)
                a_tot_g = np.einsum("gij,ij->g", D_g, a_ss)
                a_tot_base = np.einsum("gij,ij->g", D_g_ss, a_ss)

                masses_path[t] = np.sum(D_g, axis=(1, 2))
                dc_g[t] = (c_tot_g - c_tot_base) / masses
                dc_pct_g[t] = dc_g[t] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
                ds_c_g[t] = c_tot_g / C_t - c_tot_base / C_ss_tot
                da_g[t] = (a_tot_g - a_tot_base) / masses
                ds_a_g[t] = (a_tot_g / A_t - a_tot_base / A_ss_tot) if (abs(A_t) > 1e-12 and abs(A_ss_tot) > 1e-12) else 0.0

                if t + 1 < T:
                    D_g = np.array([_forward_step(D_g[g], a_ss, a_grid, pi_s) for g in range(G)])
        elif tracking in ("cross_section", "fixed_bins"):
            if tracking == "cross_section" and quantile_edges is None:
                raise ValueError(
                    "tracking='cross_section' requires quantile-based wealth partitions."
                )
            D_t = (
                _forward_step(elig[:, None] * D_ss, a_tr, a_grid, pi_s)
                + _forward_step((1.0 - elig)[:, None] * D_ss, a_ss, a_grid, pi_s)
            )
            c_tot_ss = np.einsum("gij,ij->g", W, D_ss * c_ss)
            a_tot_ss = np.einsum("gij,ij->g", W, D_ss * a_ss)
            for t in range(1, T):
                if tracking == "cross_section":
                    assert quantile_edges is not None
                    W_a_t = _quantile_edges_weights(D_t.sum(axis=1), quantile_edges)
                    W_t = np.repeat(W_a_t[:, :, None], n_s, axis=2)
                else:
                    W_t = W
                C_t = float(np.sum(D_t * c_ss))
                A_t = float(np.sum(D_t * a_ss))
                agg_dC[t] = C_t - C_ss_tot
                agg_dA[t] = A_t - A_ss_tot

                c_tot_g = np.einsum("gij,ij->g", W_t, D_t * c_ss)
                a_tot_g = np.einsum("gij,ij->g", W_t, D_t * a_ss)
                masses_path[t] = np.einsum("gij,ij->g", W_t, D_t)
                dc_g[t] = (c_tot_g - c_tot_ss) / masses
                dc_pct_g[t] = dc_g[t] / np.where(np.abs(mean_c_ss) > 1e-12, mean_c_ss, 1.0)
                ds_c_g[t] = c_tot_g / C_t - c_shares_ss
                da_g[t] = (a_tot_g - a_tot_ss) / masses
                ds_a_g[t] = (a_tot_g / A_t - a_shares_ss) if (abs(A_t) > 1e-12 and abs(A_ss_tot) > 1e-12) else 0.0
                if t + 1 < T:
                    D_t = _forward_step(D_t, a_ss, a_grid, pi_s)
        else:
            raise ValueError(f"Unknown tracking {tracking!r}; expected 'cohort', 'cross_section', or 'fixed_bins'.")

    trajectories = {
        "consumption_irf": dc_g,
        "consumption_pct_irf": dc_pct_g,
        "consumption_share_irf": ds_c_g,
        "asset_irf": da_g,
        "asset_share_irf": ds_a_g,
        "group_masses_path": masses_path,
    }
    pseudo_extracted: dict[str, Any] = {
        "agg_dC": agg_dC,
        "agg_dA": agg_dA,
        "agg_dB": None,
        "source_diag": None,
        "source_converged": bool(ss.ss_converged),
        "source_iterations": 1,
        "source_tol": tol_val,
        "source_method": f"targeted_transfer_{'nonlinear' if nonlinear else 'linear'}",
        "extra_ss": {},
        "validation_scope": "partial_equilibrium_transfer",
        "model_kind": "partial_equilibrium_transfer",
        "closure": "partial_equilibrium",
        "tax_type": "cash_on_hand_transfer",
    }
    conservation_audit, diag = _audit_conservation_and_build_diagnostics(
        pseudo_extracted,
        trajectories,
        masses,
        cal,
        conservation_tol=tol_val,
        strict_conservation=strict_conservation,
    )

    return HANKGroupIRFResult(
        horizons=np.arange(T, dtype=np.int64),
        groups=names,
        group_scheme=scheme,
        tracking=str(tracking),
        model_kind="partial_equilibrium_transfer",
        closure="partial_equilibrium",
        tax_type="cash_on_hand_transfer",
        consumption_irf=dc_g,
        consumption_pct_irf=dc_pct_g,
        consumption_share_irf=ds_c_g,
        asset_irf=da_g,
        asset_share_irf=ds_a_g,
        aggregate_consumption_irf=agg_dC,
        aggregate_asset_irf=agg_dA,
        group_masses=masses,
        group_masses_path=masses_path,
        steady_state_group_consumption=mean_c_ss,
        steady_state_group_assets=mean_a_ss,
        steady_state_group_mpc=group_mpc,
        steady_state_group_consumption_shares=c_shares_ss,
        conservation_audit=conservation_audit,
        calibration=cal,
        solver_diagnostics=diag,
        validation_scope="partial_equilibrium_transfer",
        source_result=ss,
    )


# ---------------------------------------------------------------------------
# Empirical target coercion, alignment, and scoring
# ---------------------------------------------------------------------------


def _coerce_empirical_targets(
    targets: Any,
    *,
    ci_level: float = 0.90,
) -> EmpiricalGroupTargets:
    """Normalize DistributionalLPResult, Mapping, DataFrame, or EmpiricalGroupTargets."""
    from puremacro.lp.distributional import DistributionalLPResult

    if isinstance(targets, EmpiricalGroupTargets):
        return targets

    if isinstance(targets, DistributionalLPResult):
        return EmpiricalGroupTargets(
            horizons=np.asarray(targets.horizons, dtype=np.int64),
            groups=tuple(str(b) for b in targets.bins),
            irf_matrix=targets.irf_matrix,
            se_matrix=targets.se_matrix,
            ci_lower=targets.ci_lower,
            ci_upper=targets.ci_upper,
            ci_level=float(targets.ci),
            distribution_type=str(targets.distribution_type),
            shock_name=str((targets.metadata or {}).get("shock", "")),
            metadata=dict(targets.metadata or {}),
        )

    if isinstance(targets, pd.DataFrame):
        df = targets.copy()
        if df.empty:
            raise ValueError("Empirical targets DataFrame cannot be empty")
        cols_lower = {str(c).lower(): c for c in df.columns}
        h_col = cols_lower.get("horizon") or cols_lower.get("h")
        b_col = cols_lower.get("bin") or cols_lower.get("group")
        v_col = (
            cols_lower.get("beta")
            or cols_lower.get("irf")
            or cols_lower.get("estimate")
            or cols_lower.get("value")
        )
        if h_col is not None and b_col is not None and v_col is not None:
            se_col = cols_lower.get("se")
            lo_col = cols_lower.get("ci_lower") or cols_lower.get("lower")
            hi_col = cols_lower.get("ci_upper") or cols_lower.get("upper")
            groups_ord = tuple(dict.fromkeys(df[b_col].astype(str).tolist()))
            hs_ord = np.array(sorted(set(int(x) for x in df[h_col].tolist())), dtype=np.int64)
            piv_val = df.pivot(index=h_col, columns=b_col, values=v_col).reindex(index=hs_ord, columns=groups_ord)
            se_mat = (
                df.pivot(index=h_col, columns=b_col, values=se_col).reindex(index=hs_ord, columns=groups_ord).to_numpy(dtype=np.float64)
                if se_col is not None
                else None
            )
            lo_mat = (
                df.pivot(index=h_col, columns=b_col, values=lo_col).reindex(index=hs_ord, columns=groups_ord).to_numpy(dtype=np.float64)
                if lo_col is not None
                else None
            )
            hi_mat = (
                df.pivot(index=h_col, columns=b_col, values=hi_col).reindex(index=hs_ord, columns=groups_ord).to_numpy(dtype=np.float64)
                if hi_col is not None
                else None
            )
            return EmpiricalGroupTargets(
                horizons=hs_ord,
                groups=groups_ord,
                irf_matrix=piv_val.to_numpy(dtype=np.float64),
                se_matrix=se_mat,
                ci_lower=lo_mat,
                ci_upper=hi_mat,
                ci_level=ci_level,
            )

        # Wide DataFrame fallback
        if h_col is not None:
            df = df.set_index(h_col)
        hs_arr = np.asarray(df.index, dtype=np.int64)
        se_cols = [c for c in df.columns if str(c).endswith("_se")]
        irf_cols = [c for c in df.columns if c not in se_cols]
        se_mat = None
        if se_cols:
            matched_se = [f"{c}_se" for c in irf_cols]
            if all(s in df.columns for s in matched_se):
                se_mat = df[matched_se].to_numpy(dtype=np.float64)
        return EmpiricalGroupTargets(
            horizons=hs_arr,
            groups=tuple(str(c) for c in irf_cols),
            irf_matrix=df[irf_cols].to_numpy(dtype=np.float64),
            se_matrix=se_mat,
            ci_level=ci_level,
        )

    if isinstance(targets, Mapping):
        if "irf_matrix" in targets or "irf" in targets:
            irf_raw = targets["irf_matrix"] if "irf_matrix" in targets else targets["irf"]
            irf_arr = _as_finite_array(irf_raw, "targets['irf']", ndim=2)
            H_t, G_t = irf_arr.shape
            se_raw = targets.get("se_matrix", targets.get("se"))
            groups_raw = targets.get("groups", targets.get("bins"))
            groups_tup = (
                tuple(str(g) for g in groups_raw)
                if groups_raw is not None
                else tuple(f"G{i + 1}" for i in range(G_t))
            )
            hs_raw = targets.get("horizons")
            hs_arr = np.arange(H_t, dtype=np.int64) if hs_raw is None else np.asarray(hs_raw, dtype=np.int64)
            return EmpiricalGroupTargets(
                horizons=hs_arr,
                groups=groups_tup,
                irf_matrix=irf_arr,
                se_matrix=se_raw,
                ci_lower=targets.get("ci_lower"),
                ci_upper=targets.get("ci_upper"),
                ci_level=float(targets.get("ci_level", targets.get("ci", ci_level))),
                distribution_type=str(targets.get("distribution_type", "levels")),
                shock_name=str(targets.get("shock_name", "")),
                metadata=dict(targets.get("metadata", {})),
            )
        raise ValueError(
            "Mapping targets must provide 'irf' or 'irf_matrix' (and optional 'se', 'horizons', 'groups')."
        )

    raise TypeError(
        f"Unsupported empirical_targets type {type(targets).__name__}; expected "
        "DistributionalLPResult, EmpiricalGroupTargets, pd.DataFrame, or Mapping."
    )


def _align_groups_and_horizons(
    group_irf: HANKGroupIRFResult,
    emp: EmpiricalGroupTargets,
    *,
    bin_mapping: Mapping[str, str] | None,
    horizons: Sequence[int] | None,
    warn_list: list[str],
) -> tuple[np.ndarray, list[int], list[int], tuple[str, ...], tuple[str, ...]]:
    """Resolve aligned horizon indices and empirical-bin <-> model-group bijection."""
    model_groups = group_irf.groups
    emp_bins = emp.groups
    model_lookup = {_norm_label(g): idx for idx, g in enumerate(model_groups)}
    emp_lookup = {_norm_label(b): idx for idx, b in enumerate(emp_bins)}

    if bin_mapping is not None:
        if len(bin_mapping) != len(emp_bins):
            raise ValueError(
                f"bin_mapping must cover all {len(emp_bins)} empirical bins {emp_bins}, "
                f"got {len(bin_mapping)} entries."
            )
        emp_indices: list[int] = []
        mod_indices: list[int] = []
        seen_mod: set[int] = set()
        for b_raw, g_raw in bin_mapping.items():
            b_norm, g_norm = _norm_label(b_raw), _norm_label(g_raw)
            if b_norm not in emp_lookup:
                raise ValueError(f"bin_mapping empirical bin {b_raw!r} not found in {emp_bins!r}")
            if g_norm not in model_lookup:
                raise ValueError(f"bin_mapping model group {g_raw!r} not found in {model_groups!r}")
            m_idx = model_lookup[g_norm]
            if m_idx in seen_mod:
                raise ValueError(f"bin_mapping maps multiple empirical bins onto model group {g_raw!r}")
            seen_mod.add(m_idx)
            emp_indices.append(emp_lookup[b_norm])
            mod_indices.append(m_idx)
    else:
        if len(emp_bins) != len(model_groups):
            raise ValueError(
                f"Incompatible group counts: empirical targets have {len(emp_bins)} bins "
                f"{emp_bins!r}, while HANK model has {len(model_groups)} groups {model_groups!r}. "
                "Provide matching group count or an explicit bin_mapping."
            )
        if all(_norm_label(b) in model_lookup for b in emp_bins):
            emp_indices = list(range(len(emp_bins)))
            mod_indices = [model_lookup[_norm_label(b)] for b in emp_bins]
        else:
            emp_indices = list(range(len(emp_bins)))
            mod_indices = list(range(len(model_groups)))
            pair_txt = ", ".join(f"{b}->{g}" for b, g in zip(emp_bins, model_groups))
            warn_list.append(f"Aligned empirical bins to HANK groups by order ({pair_txt}).")

    aligned_groups = tuple(model_groups[i] for i in mod_indices)
    aligned_bins = tuple(emp_bins[i] for i in emp_indices)

    model_h_set = set(int(h) for h in group_irf.horizons.tolist())
    filter_set = set(int(h) for h in horizons) if horizons is not None else None
    common_h = [
        int(h)
        for h in emp.horizons.tolist()
        if int(h) in model_h_set and (filter_set is None or int(h) in filter_set)
    ]
    if not common_h:
        raise ValueError(
            "No overlapping horizons between HANK model "
            f"[{int(group_irf.horizons[0])}..{int(group_irf.horizons[-1])}] "
            f"and empirical targets {emp.horizons.tolist()!r}."
        )
    if len(common_h) < len(emp.horizons):
        warn_list.append(
            f"Aligned {len(common_h)} of {len(emp.horizons)} empirical horizons within HANK horizon [0..{int(group_irf.horizons[-1])}]."
        )

    return np.asarray(common_h, dtype=np.int64), emp_indices, mod_indices, aligned_groups, aligned_bins


def _resolve_response_type(
    response_type: str,
    emp: EmpiricalGroupTargets,
    warn_list: list[str],
) -> str:
    """Resolve model IRF scale ('level', 'percent', 'share', 'asset_level', 'asset_share')."""
    key = str(response_type).strip().lower()
    if key == "auto":
        dist = str(emp.distribution_type).strip().lower()
        if dist in ("shares", "share"):
            return "share"
        if dist in ("percent", "pct"):
            return "percent"
        if dist in ("assets", "asset", "asset_level"):
            return "asset_level"
        if dist in ("asset_share", "asset_shares"):
            return "asset_share"
        if dist == "quantiles":
            warn_list.append(
                "Quantile LP targets aligned against HANK group consumption levels ('level'); "
                "specify response_type='percent' if targets are log/percentage responses."
            )
        return "level"

    aliases = {
        "level": "level",
        "levels": "level",
        "consumption": "level",
        "consumption_level": "level",
        "percent": "percent",
        "pct": "percent",
        "consumption_pct": "percent",
        "share": "share",
        "shares": "share",
        "consumption_share": "share",
        "asset": "asset_level",
        "assets": "asset_level",
        "asset_level": "asset_level",
        "asset_share": "asset_share",
    }
    resolved = aliases.get(key)
    if resolved is None:
        raise ValueError(
            f"Unknown response_type {response_type!r}; expected 'auto', 'level', 'percent', "
            "'share', 'asset_level', or 'asset_share'."
        )
    return resolved


def _compute_gradient_metrics(
    aligned_h: np.ndarray,
    aligned_groups: tuple[str, ...],
    model_mat: np.ndarray,
    emp_mat: np.ndarray,
    *,
    low_group: str | None,
    high_group: str | None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Compute cross-group inequality-gradient contrast and cross-group dispersion."""
    if low_group is None:
        low_idx = 0
    else:
        if low_group not in aligned_groups:
            raise ValueError(f"low_group {low_group!r} not in aligned groups {aligned_groups!r}")
        low_idx = aligned_groups.index(low_group)
    if high_group is None:
        high_idx = len(aligned_groups) - 1
    else:
        if high_group not in aligned_groups:
            raise ValueError(f"high_group {high_group!r} not in aligned groups {aligned_groups!r}")
        high_idx = aligned_groups.index(high_group)

    low_name = aligned_groups[low_idx]
    high_name = aligned_groups[high_idx]

    low_m = model_mat[:, low_idx]
    high_m = model_mat[:, high_idx]
    diff_m = low_m - high_m

    low_e = emp_mat[:, low_idx]
    high_e = emp_mat[:, high_idx]
    diff_e = low_e - high_e

    disp_m = np.std(model_mat, axis=1)
    disp_e = np.std(emp_mat, axis=1)

    grad_diff = diff_m - diff_e
    grad_rmse = float(np.sqrt(np.mean(grad_diff ** 2)))
    grad_mae = float(np.mean(np.abs(grad_diff)))
    grad_sign = float(np.mean([_sign_match(float(a), float(b)) for a, b in zip(diff_m, diff_e)]))

    cum_low_m = float(np.sum(low_m))
    cum_high_m = float(np.sum(high_m))
    cum_low_e = float(np.sum(low_e))
    cum_high_e = float(np.sum(high_e))
    mean_disp_m = float(np.mean(disp_m))
    mean_disp_e = float(np.mean(disp_e))

    gradient_metrics: dict[str, Any] = {
        "low_group": low_name,
        "high_group": high_name,
        "gradient_impact_diff_model": float(diff_m[0]),
        "gradient_impact_diff_empirical": float(diff_e[0]),
        "gradient_cumulative_diff_model": float(np.sum(diff_m)),
        "gradient_cumulative_diff_empirical": float(np.sum(diff_e)),
        "gradient_impact_ratio_model": _safe_ratio(float(low_m[0]), float(high_m[0])),
        "gradient_impact_ratio_empirical": _safe_ratio(float(low_e[0]), float(high_e[0])),
        "gradient_cumulative_ratio_model": _safe_ratio(cum_low_m, cum_high_m),
        "gradient_cumulative_ratio_empirical": _safe_ratio(cum_low_e, cum_high_e),
        "gradient_diff_rmse": grad_rmse,
        "gradient_diff_mae": grad_mae,
        "gradient_sign_concordance": grad_sign,
        "dispersion_mean_model": mean_disp_m,
        "dispersion_mean_empirical": mean_disp_e,
        "dispersion_impact_range_model": float(np.max(model_mat[0]) - np.min(model_mat[0])),
        "dispersion_impact_range_empirical": float(np.max(emp_mat[0]) - np.min(emp_mat[0])),
        "dispersion_ratio": _safe_ratio(mean_disp_m, mean_disp_e),
    }

    grad_table = pd.DataFrame(
        {
            "horizon": aligned_h,
            "low_model": low_m,
            "high_model": high_m,
            "diff_model": diff_m,
            "low_empirical": low_e,
            "high_empirical": high_e,
            "diff_empirical": diff_e,
            "dispersion_model": disp_m,
            "dispersion_empirical": disp_e,
        }
    ).set_index("horizon")

    return gradient_metrics, grad_table


def bridge_hank_distributional(
    model_or_group_irf: Any,
    empirical_targets: Any,
    *,
    groups: str | int | Sequence[float] | Mapping[str, np.ndarray] = "quintiles",
    group_labels: Sequence[str] | None = None,
    htm_definition: Literal["constrained", "bottom_quartile"] = "constrained",
    tracking: Literal["cohort", "cross_section", "fixed_bins"] = "cohort",
    response_type: Literal[
        "auto", "level", "percent", "share", "asset_level", "asset_share"
    ] = "auto",
    shock_scale: float = 1.0,
    bin_mapping: Mapping[str, str] | None = None,
    horizons: Sequence[int] | None = None,
    low_group: str | None = None,
    high_group: str | None = None,
    ci_level: float = 0.90,
    conservation_tol: float = _DEFAULT_CONSERVATION_TOL,
    strict_conservation: bool = True,
    allow_unconverged: bool = False,
) -> HANKDistributionalBridgeResult:
    """Align and score HANK household-group IRFs against empirical distributional LPs.

    Parameters
    ----------
    model_or_group_irf : HANKGroupIRFResult | FiscalHANKResult | NonlinearHANKResult | SequenceSpaceHANKResult
        Solved HANK model result or pre-extracted :class:`HANKGroupIRFResult`.
    empirical_targets : DistributionalLPResult | EmpiricalGroupTargets | pd.DataFrame | Mapping
        Empirical distributional/quantile LP estimates or observed group targets + SEs.
    groups, group_labels, htm_definition, tracking :
        Used when ``model_or_group_irf`` is a raw HANK solver result rather than
        an already-extracted :class:`HANKGroupIRFResult`.
    response_type : {'auto', 'level', 'percent', 'share', 'asset_level', 'asset_share'}
        Unit/metric alignment against empirical targets. ``'auto'`` selects
        ``'share'`` for share-type ``DistributionalLPResult`` and ``'level'``
        otherwise.
    shock_scale : float, default 1.0
        Explicit non-zero multiplier applied to model group IRFs (no silent scaling).
    bin_mapping : Mapping[str, str] or None, optional
        Optional mapping from empirical bin names to model group names.
    horizons : Sequence[int] or None, optional
        Optional subset of horizons to include in alignment scoring.
    low_group, high_group : str or None, optional
        Group labels defining the inequality-gradient contrast (defaults to
        lowest-wealth group ``groups[0]`` vs. highest-wealth group ``groups[-1]``).
    ci_level : float, default 0.90
        Fallback confidence level when empirical DataFrame/Mapping omits bands.
    conservation_tol : float, default 1e-10
        Cross-sectional conservation audit tolerance.
    strict_conservation : bool, default True
        Fail closed with ``ValueError`` if conservation tolerance is violated.
    allow_unconverged : bool, default False
        Refuse unconverged HANK results when ``False``.
    """
    from scipy.stats import chi2

    scale_val = float(shock_scale)
    if not (math.isfinite(scale_val) and scale_val != 0.0):
        raise ValueError(f"shock_scale must be a finite non-zero float, got {shock_scale!r}")

    if isinstance(model_or_group_irf, HANKGroupIRFResult):
        group_irf = model_or_group_irf
    else:
        group_irf = extract_hank_group_irf(
            model_or_group_irf,
            groups=groups,
            group_labels=group_labels,
            htm_definition=htm_definition,
            tracking=tracking,
            conservation_tol=conservation_tol,
            strict_conservation=strict_conservation,
            allow_unconverged=allow_unconverged,
        )

    emp = _coerce_empirical_targets(empirical_targets, ci_level=ci_level)
    warn_list: list[str] = list(group_irf.support_status.warnings)
    warn_list.append(_scope_note(group_irf.validation_scope, "en"))

    aligned_h, emp_idx, mod_idx, aligned_groups, aligned_bins = _align_groups_and_horizons(
        group_irf,
        emp,
        bin_mapping=bin_mapping,
        horizons=horizons,
        warn_list=warn_list,
    )
    resolved_resp = _resolve_response_type(response_type, emp, warn_list)

    mod_full, _, _ = group_irf._select_matrix(resolved_resp)
    h_to_mod_row = {int(h): i for i, h in enumerate(group_irf.horizons.tolist())}
    h_to_emp_row = {int(h): i for i, h in enumerate(emp.horizons.tolist())}

    mod_rows = [h_to_mod_row[int(h)] for h in aligned_h.tolist()]
    emp_rows = [h_to_emp_row[int(h)] for h in aligned_h.tolist()]

    model_mat = scale_val * mod_full[np.ix_(mod_rows, mod_idx)]
    emp_mat = emp.irf_matrix[np.ix_(emp_rows, emp_idx)]

    H_al, G_al = model_mat.shape
    if emp.se_matrix is not None:
        se_mat = emp.se_matrix[np.ix_(emp_rows, emp_idx)]
    else:
        se_mat = np.full((H_al, G_al), np.nan, dtype=np.float64)
        warn_list.append("Empirical standard errors not provided; t-statistics and Wald chi2 are NaN.")

    if emp.ci_lower is not None and emp.ci_upper is not None:
        lo_mat = emp.ci_lower[np.ix_(emp_rows, emp_idx)]
        hi_mat = emp.ci_upper[np.ix_(emp_rows, emp_idx)]
    else:
        lo_mat = np.full((H_al, G_al), np.nan, dtype=np.float64)
        hi_mat = np.full((H_al, G_al), np.nan, dtype=np.float64)

    aligned_rows: list[dict[str, Any]] = []
    for i_h, h_val in enumerate(aligned_h.tolist()):
        for j_g, (g_name, b_name, m_orig_idx) in enumerate(zip(aligned_groups, aligned_bins, mod_idx)):
            m_v = float(model_mat[i_h, j_g])
            e_v = float(emp_mat[i_h, j_g])
            se_v = float(se_mat[i_h, j_g])
            lo_v = float(lo_mat[i_h, j_g])
            hi_v = float(hi_mat[i_h, j_g])
            diff_v = m_v - e_v
            t_v = diff_v / se_v if (math.isfinite(se_v) and se_v > 0.0) else float("nan")
            in_band = bool(math.isfinite(lo_v) and math.isfinite(hi_v) and (lo_v <= m_v <= hi_v))
            aligned_rows.append(
                {
                    "horizon": int(h_val),
                    "group": g_name,
                    "empirical_bin": b_name,
                    "model": m_v,
                    "empirical": e_v,
                    "se": se_v,
                    "ci_lower": lo_v,
                    "ci_upper": hi_v,
                    "diff": diff_v,
                    "t_stat": t_v,
                    "abs_t_stat": abs(t_v) if math.isfinite(t_v) else float("nan"),
                    "sign_match": _sign_match(m_v, e_v),
                    "within_band": in_band,
                    "population_mass": float(group_irf.group_masses[m_orig_idx]),
                }
            )
    aligned_table = pd.DataFrame(aligned_rows)

    diff_mat = model_mat - emp_mat
    rmse = float(np.sqrt(np.mean(diff_mat ** 2)))
    mae = float(np.mean(np.abs(diff_mat)))
    sign_conc = float(aligned_table["sign_match"].mean())

    has_bands = np.isfinite(lo_mat) & np.isfinite(hi_mat)
    if np.any(has_bands):
        in_band_arr = (model_mat >= lo_mat) & (model_mat <= hi_mat)
        band_cov = float(np.mean(in_band_arr[has_bands]))
    else:
        band_cov = float("nan")

    valid_se = np.isfinite(se_mat) & (se_mat > 0.0)
    if np.any(valid_se):
        t_vals = diff_mat[valid_se] / se_mat[valid_se]
        weighted_rmse = float(np.sqrt(np.mean(t_vals ** 2)))
        chi2_stat = float(np.sum(t_vals ** 2))
        chi2_df = int(np.sum(valid_se))
        chi2_pval = float(chi2.sf(chi2_stat, df=chi2_df))
        mean_abs_t = float(np.mean(np.abs(t_vals)))
        max_abs_t = float(np.max(np.abs(t_vals)))
        if chi2_df > 1:
            warn_list.append(
                "Wald chi2 uses diagonal pointwise LP variances (cross-horizon/bin covariance unavailable)."
            )
    else:
        weighted_rmse = float("nan")
        chi2_stat = float("nan")
        chi2_df = 0
        chi2_pval = float("nan")
        mean_abs_t = float("nan")
        max_abs_t = float("nan")

    per_group_rows: list[dict[str, Any]] = []
    corrs_valid: list[float] = []
    for j_g, (g_name, b_name, m_orig_idx) in enumerate(zip(aligned_groups, aligned_bins, mod_idx)):
        m_vec = model_mat[:, j_g]
        e_vec = emp_mat[:, j_g]
        d_vec = m_vec - e_vec
        se_vec = se_mat[:, j_g]
        v_se = np.isfinite(se_vec) & (se_vec > 0.0)
        b_mask = has_bands[:, j_g]

        std_m = float(np.std(m_vec))
        std_e = float(np.std(e_vec))
        if len(aligned_h) >= 2 and std_m > 1e-12 and std_e > 1e-12:
            corr = float(np.corrcoef(m_vec, e_vec)[0, 1])
            corr = max(-1.0, min(1.0, corr))
            corrs_valid.append(corr)
        elif np.allclose(m_vec, e_vec, atol=1e-10):
            corr = 1.0
            corrs_valid.append(corr)
        else:
            corr = float("nan")

        p_mod = int(aligned_h[int(np.argmax(np.abs(m_vec)))])
        p_emp = int(aligned_h[int(np.argmax(np.abs(e_vec)))])
        abs_t_g = np.abs(d_vec[v_se] / se_vec[v_se]) if np.any(v_se) else np.array([], dtype=np.float64)
        signs_g = [_sign_match(float(x), float(y)) for x, y in zip(m_vec, e_vec)]

        per_group_rows.append(
            {
                "group": g_name,
                "empirical_bin": b_name,
                "population_mass": float(group_irf.group_masses[m_orig_idx]),
                "steady_state_mpc": float(group_irf.steady_state_group_mpc[m_orig_idx]),
                "steady_state_wealth": float(group_irf.steady_state_group_assets[m_orig_idx]),
                "steady_state_consumption": float(group_irf.steady_state_group_consumption[m_orig_idx]),
                "rmse": float(np.sqrt(np.mean(d_vec ** 2))),
                "mae": float(np.mean(np.abs(d_vec))),
                "bias": float(np.mean(d_vec)),
                "sign_concordance": float(np.mean(signs_g)),
                "band_coverage": float(np.mean((m_vec[b_mask] >= lo_mat[b_mask, j_g]) & (m_vec[b_mask] <= hi_mat[b_mask, j_g]))) if np.any(b_mask) else float("nan"),
                "mean_abs_t_stat": float(np.mean(abs_t_g)) if abs_t_g.size else float("nan"),
                "max_abs_t_stat": float(np.max(abs_t_g)) if abs_t_g.size else float("nan"),
                "correlation": corr,
                "peak_horizon_model": p_mod,
                "peak_horizon_empirical": p_emp,
                "peak_timing_gap": p_emp - p_mod,
                "impact_model": float(m_vec[0]),
                "impact_empirical": float(e_vec[0]),
                "cumulative_model": float(np.sum(m_vec)),
                "cumulative_empirical": float(np.sum(e_vec)),
            }
        )
    per_group_df = pd.DataFrame(per_group_rows).set_index("group")
    mean_corr = float(np.mean(corrs_valid)) if corrs_valid else float("nan")

    gradient_metrics, grad_table = _compute_gradient_metrics(
        aligned_h,
        aligned_groups,
        model_mat,
        emp_mat,
        low_group=low_group,
        high_group=high_group,
    )

    cal = {
        **group_irf.calibration,
        "aligned_groups": list(aligned_groups),
        "aligned_empirical_bins": list(aligned_bins),
        "response_type": resolved_resp,
        "shock_scale": scale_val,
    }

    return HANKDistributionalBridgeResult(
        group_irf=group_irf,
        aligned_horizons=aligned_h,
        groups=aligned_groups,
        empirical_bins=aligned_bins,
        response_type=resolved_resp,
        shock_scale=scale_val,
        model_irf=model_mat,
        empirical_irf=emp_mat,
        empirical_se=se_mat,
        ci_lower=lo_mat,
        ci_upper=hi_mat,
        ci_level=float(emp.ci_level),
        aligned_table=aligned_table,
        per_group_metrics=per_group_df,
        gradient_metrics=gradient_metrics,
        gradient_table=grad_table,
        rmse=rmse,
        mae=mae,
        weighted_rmse=weighted_rmse,
        chi2_stat=chi2_stat,
        chi2_df=chi2_df,
        chi2_pvalue=chi2_pval,
        sign_concordance=sign_conc,
        band_coverage=band_cov,
        mean_abs_t_stat=mean_abs_t,
        max_abs_t_stat=max_abs_t,
        mean_trajectory_correlation=mean_corr,
        calibration=cal,
        conservation_audit=dict(group_irf.conservation_audit),
        solver_diagnostics=group_irf.solver_diagnostics,
        warnings=tuple(dict.fromkeys(warn_list)),
        validation_scope=group_irf.validation_scope,
    )


def score_hank_distributional(
    model_or_group_irf: Any,
    empirical_targets: Any,
    **kwargs: Any,
) -> HANKDistributionalBridgeResult:
    """Alias for :func:`bridge_hank_distributional`."""
    return bridge_hank_distributional(model_or_group_irf, empirical_targets, **kwargs)
