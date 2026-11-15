"""Dynamic trade reallocation incidence and OECD ICIO calibration bridge.

This module bridges the two-country, two-sector dynamic reallocation model
(:mod:`puremacro.trade.reallocation`) with:

1. **Deterministic OECD ICIO calibration** (:func:`calibrate_reallocation_from_icio`),
   aggregating static multi-region, multi-sector input-output accounts into a
   validated :class:`~puremacro.trade.reallocation.ReallocationEconomy` while
   conserving value added and expenditure flows and disclosing every dynamic
   and behavioral parameter not identified by single-year IO tables.
2. **Benchmark-normalized transition household incidence**
   (:func:`evaluate_transition_household_incidence`), decomposing a converged
   :class:`~puremacro.trade.reallocation.ReallocationTransitionResult` across
   household groups with heterogeneous sector-labor weights, non-labor/rebate
   incidence, and demand systems calibrated via :mod:`puremacro.trade.household`
   (``fixed_baskets``, ``cobb_douglas``, ``stone_geary``, or ``ces``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from puremacro._numeric_contracts import freeze as _freeze_metadata, readonly_array, real_scalar
from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.solvers.diagnostics import SolverDiagnosticRecord, make_solver_diagnostic_record

from .data import ICIOData, RawIOData, load_icio_data
from .dynamic.welfare import consumption_equivalent_welfare
from .household import HouseholdPreferences, calibrate_household, compute_household_welfare
from .reallocation import (
    ReallocationEconomy,
    ReallocationSteadyState,
    ReallocationTransitionResult,
    audit_reallocation,
    reallocation_stability,
    solve_reallocation_steady_state,
)
from .scenarios import CompositeTariffScenario, TariffScenario

__all__ = [
    "TransitionError",
    "ICIOCalibrationAudit",
    "ReallocationCalibrationResult",
    "HouseholdGroupSpec",
    "TransitionHouseholdIncidenceResult",
    "calibrate_reallocation_from_icio",
    "reallocation_tariffs_from_scenarios",
    "make_sector_household_groups",
    "evaluate_transition_household_incidence",
]

_VALID_LANGUAGES: tuple[str, str] = ("en", "es")
_VALID_RULES: tuple[str, ...] = ("fixed_baskets", "cobb_douglas", "stone_geary", "ces")
_VALID_EXPENDITURE_BASES: tuple[str, ...] = ("final_demand", "consumption", "total_absorption")

# Same 22 OECD ICIO 45-sector goods codes used in DynamicAccounts, plus single-letter ISIC A-C.
_OECD_45_GOODS: frozenset[str] = frozenset({
    "A01_02", "A03", "B05_06", "B07_08", "B09", "C10T12", "C13T15", "C16", "C17_18",
    "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29",
    "C30", "C31T33",
})


class TransitionError(ValueError):
    """Raised when a dynamic trade transition is unconverged, unaudited, or invalid."""


def _validate_lang(lang: str) -> Literal["en", "es"]:
    if lang not in _VALID_LANGUAGES:
        raise ValueError(f"lang must be 'en' or 'es', got {lang!r}")
    return lang  # type: ignore[return-value]


def _is_merchandise_sector(code: str) -> bool:
    s = str(code).strip()
    if s in _OECD_45_GOODS or s in {"A", "B", "C"}:
        return True
    if len(s) >= 2 and s[0] in {"A", "B", "C"} and s[1].isdigit() and s != "C33":
        return True
    return False


def _validate_nonneg_vector(value: Any, name: str, length: int) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        arr = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric sequence of length {length}") from exc
    if arr.shape != (length,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite with shape ({length},), got {arr.shape}")
    if (arr < 0).any():
        raise ValueError(f"{name} must be nonnegative")
    return readonly_array(arr)


def _check_matrix_2x2(value: Any, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        arr = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array with shape (2, 2)") from exc
    if arr.shape != (2, 2) or not np.isfinite(arr).all() or (arr <= 0).any():
        raise ValueError(f"{name} must be strictly positive and finite with shape (2, 2)")
    return readonly_array(arr)


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Mapping):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_json_safe(v) for v in obj]
    return obj


# ---------------------------------------------------------------------------
# ICIO Extraction and 2x2 Aggregation Helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _ExtractedIO:
    source: str
    country_codes: tuple[str, ...]
    sector_codes: tuple[str, ...]
    fd_codes: tuple[str, ...]
    intermediate: np.ndarray  # (nc, ns, nc, ns)
    final_demand: np.ndarray  # (nc, ns, nc, nfd)
    labor_va: np.ndarray      # (nc, ns)
    capital_va: np.ndarray    # (nc, ns)
    tls: np.ndarray           # (nc, ns)
    tfd: np.ndarray           # (nc, nfd)
    gross_output: np.ndarray  # (nc, ns)
    factor_split_note: str


def _extract_io_arrays(
    icio: Any,
    country_codes: Sequence[str] | None,
    sector_codes: Sequence[str] | None,
    fd_codes: Sequence[str] | None,
) -> _ExtractedIO:
    if icio is None:
        icio = load_icio_data(return_structured=True)

    if isinstance(icio, ICIOData):
        c_codes = tuple(str(c) for c in icio.country_codes)
        s_codes = tuple(str(s) for s in icio.sector_codes)
        f_codes = tuple(str(f) for f in icio.fd_codes)
        nc, ns, nf = len(c_codes), len(s_codes), len(f_codes)
        Z = np.asarray(icio.intermediate_matrix, dtype=float).reshape(nc, ns, nc, ns)
        F = np.asarray(icio.final_demand_matrix, dtype=float).reshape(nc, ns, nc, nf)
        va_l = np.asarray(icio.labor_va, dtype=float).reshape(nc, ns)
        va_k = np.asarray(icio.capital_va, dtype=float).reshape(nc, ns)
        tls = np.asarray(icio.net_taxes_intermediate, dtype=float).reshape(nc, ns)
        tfd = np.asarray(icio.net_taxes_final_demand, dtype=float).reshape(nc, nf)
        out = np.asarray(icio.gross_output, dtype=float).reshape(nc, ns)
        meta = dict(icio.metadata or {})
        factor_note = str(meta.get(
            "factor_detail",
            "ICIOData labor_va/capital_va rows (bundled OECD 2023 table applies a 2/3 labor, 1/3 capital split)",
        ))
        return _ExtractedIO(
            source=str(meta.get("source", "ICIOData")),
            country_codes=c_codes,
            sector_codes=s_codes,
            fd_codes=f_codes,
            intermediate=Z,
            final_demand=F,
            labor_va=va_l,
            capital_va=va_k,
            tls=tls,
            tfd=tfd,
            gross_output=out,
            factor_split_note=factor_note,
        )

    if isinstance(icio, RawIOData):
        c_codes = tuple(str(c) for c in icio.countries)
        s_codes = tuple(str(s) for s in icio.sectors)
        f_codes = tuple(str(f) for f in icio.fd_categories)
        nc, ns, nf = len(c_codes), len(s_codes), len(f_codes)
        Z = np.asarray(icio.intermediate_matrix, dtype=float).reshape(nc, ns, nc, ns)
        F = np.asarray(icio.final_demand_matrix, dtype=float).reshape(nc, ns, nc, nf)
        va_raw = np.asarray(icio.value_added, dtype=float)
        va = va_raw.sum(axis=0) if va_raw.ndim == 2 else va_raw
        va = va.reshape(nc, ns)
        va_l = (2.0 / 3.0) * va
        va_k = (1.0 / 3.0) * va
        tls = np.asarray(icio.taxes_less_subsidies, dtype=float).reshape(nc, ns)
        tfd = (
            np.zeros((nc, nf), dtype=float)
            if icio.taxes_less_subsidies_fd is None
            else np.asarray(icio.taxes_less_subsidies_fd, dtype=float).reshape(nc, nf)
        )
        out = np.asarray(icio.gross_output, dtype=float).reshape(nc, ns)
        return _ExtractedIO(
            source=f"RawIOData:{icio.source or 'custom'}",
            country_codes=c_codes,
            sector_codes=s_codes,
            fd_codes=f_codes,
            intermediate=Z,
            final_demand=F,
            labor_va=va_l,
            capital_va=va_k,
            tls=tls,
            tfd=tfd,
            gross_output=out,
            factor_split_note="RawIOData has scalar VA row; stylized 2/3 labor, 1/3 capital split applied",
        )

    if country_codes is None or sector_codes is None:
        raise TypeError(
            "icio must be ICIOData, RawIOData, or a raw calibration matrix paired with "
            "explicit country_codes and sector_codes"
        )
    f_codes = ("C", "I", "Cx") if fd_codes is None else tuple(str(f) for f in fd_codes)
    structured = ICIOData(
        matrix=np.asarray(icio, dtype=float),
        country_codes=tuple(str(c) for c in country_codes),
        sector_codes=tuple(str(s) for s in sector_codes),
        fd_codes=f_codes,
    )
    return _extract_io_arrays(structured, None, None, None)


def _resolve_partition(
    codes: tuple[str, ...],
    groups: Mapping[str, Sequence[str]] | None,
    *,
    kind: str,
) -> tuple[tuple[str, str], dict[str, tuple[str, ...]], list[int]]:
    if groups is None:
        if len(codes) == 2:
            c0, c1 = codes[0], codes[1]
            mapping: Mapping[str, Sequence[str]] = {c0: (c0,), c1: (c1,)}
        elif kind == "country" and "USA" in codes and len(codes) > 2:
            mapping = {"USA": ("USA",), "ROW": tuple(c for c in codes if c != "USA")}
        elif kind == "sector":
            goods = tuple(s for s in codes if _is_merchandise_sector(s))
            services = tuple(s for s in codes if not _is_merchandise_sector(s))
            if not goods or not services:
                raise ValueError(
                    f"Cannot build default 2-sector mapping from {codes}; pass sector_groups explicitly"
                )
            mapping = {"Tradable": goods, "Services": services}
        else:
            raise ValueError(
                f"Cannot infer default 2-{kind} partition for {codes}; pass {kind}_groups explicitly"
            )
    else:
        mapping = groups

    if len(mapping) != 2:
        raise ValueError(f"{kind}_groups must define exactly 2 coarse groups, got {len(mapping)}")

    coarse_names: list[str] = []
    normalized: dict[str, tuple[str, ...]] = {}
    fine_to_coarse: dict[str, int] = {}
    code_set = set(codes)

    for g_idx, (name_raw, members_raw) in enumerate(mapping.items()):
        name = str(name_raw).strip()
        if not name or name in normalized:
            raise ValueError(f"{kind}_groups keys must be distinct non-empty strings")
        members: tuple[str, ...]
        if isinstance(members_raw, str):
            members = (members_raw,)
        else:
            members = tuple(str(m) for m in members_raw)
        if not members:
            raise ValueError(f"{kind}_groups[{name!r}] cannot be empty")
        for m in members:
            if m not in code_set:
                raise ValueError(f"Unknown {kind} code {m!r} in {kind}_groups[{name!r}]")
            if m in fine_to_coarse:
                raise ValueError(f"Duplicate {kind} code {m!r} across {kind}_groups")
            fine_to_coarse[m] = g_idx
        coarse_names.append(name)
        normalized[name] = members

    missing = [c for c in codes if c not in fine_to_coarse]
    if missing:
        raise ValueError(
            f"{kind}_groups must partition all {kind} codes; unmapped codes: {missing}"
        )
    idx_vec = [fine_to_coarse[c] for c in codes]
    return (coarse_names[0], coarse_names[1]), normalized, idx_vec


def _aggregate_2x2(io: _ExtractedIO, c_idx: list[int], s_idx: list[int]) -> dict[str, np.ndarray]:
    nf = len(io.fd_codes)
    z_agg = np.zeros((2, 2, 2, 2), dtype=float)
    f_agg = np.zeros((2, 2, 2, nf), dtype=float)
    va_l_agg = np.zeros((2, 2), dtype=float)
    va_k_agg = np.zeros((2, 2), dtype=float)
    tls_agg = np.zeros((2, 2), dtype=float)
    tfd_agg = np.zeros((2, nf), dtype=float)
    out_agg = np.zeros((2, 2), dtype=float)

    c_arr = np.asarray(c_idx, dtype=int)
    s_arr = np.asarray(s_idx, dtype=int)
    for src_c in range(len(io.country_codes)):
        gc = int(c_arr[src_c])
        for dst_c in range(len(io.country_codes)):
            gdc = int(c_arr[dst_c])
            for src_s in range(len(io.sector_codes)):
                gs = int(s_arr[src_s])
                f_agg[gc, gs, gdc, :] += io.final_demand[src_c, src_s, dst_c, :]
                for dst_s in range(len(io.sector_codes)):
                    gds = int(s_arr[dst_s])
                    z_agg[gc, gs, gdc, gds] += io.intermediate[src_c, src_s, dst_c, dst_s]

    for c_i in range(len(io.country_codes)):
        gc = int(c_arr[c_i])
        tfd_agg[gc, :] += io.tfd[c_i, :]
        for s_i in range(len(io.sector_codes)):
            gs = int(s_arr[s_i])
            va_l_agg[gc, gs] += io.labor_va[c_i, s_i]
            va_k_agg[gc, gs] += io.capital_va[c_i, s_i]
            tls_agg[gc, gs] += io.tls[c_i, s_i]
            out_agg[gc, gs] += io.gross_output[c_i, s_i]

    return {
        "Z": z_agg,
        "F": f_agg,
        "VA_L": va_l_agg,
        "VA_K": va_k_agg,
        "TLS": tls_agg,
        "TFD": tfd_agg,
        "output": out_agg,
    }


# ---------------------------------------------------------------------------
# ICIO Calibration Result Containers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ICIOCalibrationAudit:
    """Transparent provenance and non-identification audit for ICIO -> ReallocationEconomy."""

    source: str
    country_groups: dict[str, tuple[str, ...]]
    sector_groups: dict[str, tuple[str, ...]]
    source_countries: tuple[str, ...]
    source_sectors: tuple[str, ...]
    expenditure_basis: str
    pooled_capital_share: float
    empirical_cell_capital_shares: np.ndarray
    capital_share_source: str
    aggregated_va: np.ndarray
    aggregated_labor_va: np.ndarray
    aggregated_capital_va: np.ndarray
    aggregated_output: np.ndarray
    aggregated_final_demand: np.ndarray
    aggregated_consumption: np.ndarray | None
    aggregated_intermediate: np.ndarray
    aggregated_tls: np.ndarray
    aggregated_tfd: np.ndarray
    fine_expenditure_flows: np.ndarray
    world_va_total: float
    world_labor_va_total: float
    labor_scale_factor: float
    empirical_va_shares: np.ndarray
    steady_state_va_shares: np.ndarray
    va_share_autarky_gap: float
    stylized_assumptions: dict[str, str]
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "country_groups", {k: tuple(v) for k, v in self.country_groups.items()})
        object.__setattr__(self, "sector_groups", {k: tuple(v) for k, v in self.sector_groups.items()})
        for name in (
            "empirical_cell_capital_shares",
            "aggregated_va",
            "aggregated_labor_va",
            "aggregated_capital_va",
            "aggregated_output",
            "aggregated_final_demand",
            "aggregated_consumption",
            "aggregated_intermediate",
            "aggregated_tls",
            "aggregated_tfd",
            "fine_expenditure_flows",
            "empirical_va_shares",
            "steady_state_va_shares",
        ):
            val = getattr(self, name)
            if val is not None:
                object.__setattr__(self, name, readonly_array(np.array(val, dtype=float, copy=True)))
        object.__setattr__(self, "stylized_assumptions", dict(self.stylized_assumptions))
        object.__setattr__(self, "warnings", tuple(self.warnings))


@dataclass(frozen=True)
class ReallocationCalibrationResult:
    """Validated 2x2 dynamic reallocation economy calibrated from an ICIO table."""

    economy: ReallocationEconomy
    steady_state: ReallocationSteadyState
    stability: dict[str, Any]
    audit: ICIOCalibrationAudit
    diagnostic: SolverDiagnosticRecord
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stability", _freeze_metadata(self.stability))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def summary(self, lang: str = "en") -> pd.DataFrame:
        """Return bilingual calibration & steady-state diagnostic summary table."""
        lcode = _validate_lang(lang)
        econ = self.economy
        aud = self.audit
        rows: list[dict[str, Any]] = []
        for c_idx, c_name in enumerate(econ.countries):
            for s_idx, s_name in enumerate(econ.sectors):
                rows.append({
                    "Region" if lcode == "en" else "Región": c_name,
                    "Sector": s_name,
                    "Expenditure_Weight" if lcode == "en" else "Peso_Gasto": float(econ.sector_weights[c_idx, s_idx]),
                    "Home_Share" if lcode == "en" else "Cuota_Doméstica": float(econ.home_shares[c_idx, s_idx]),
                    "ICIO_VA_Share" if lcode == "en" else "Cuota_VA_ICIO": float(aud.empirical_va_shares[c_idx, s_idx]),
                    "SS_VA_Share" if lcode == "en" else "Cuota_VA_EE": float(aud.steady_state_va_shares[c_idx, s_idx]),
                    "Labor_Supply" if lcode == "en" else "Oferta_Laboral": float(econ.labor_supply[c_idx]),
                    "Capital_Share" if lcode == "en" else "Cuota_Capital": float(econ.capital_share),
                })
        return pd.DataFrame(rows)

    def to_dataframe(self, lang: str = "en") -> pd.DataFrame:
        """Alias for :meth:`summary`."""
        return self.summary(lang=lang)

    def to_frame(self, lang: str = "en") -> pd.DataFrame:
        """Alias for :meth:`summary`."""
        return self.summary(lang=lang)

    def to_markdown(self, lang: str = "en", **kwargs: Any) -> str:
        """Render the calibration summary table as Markdown."""
        return df_to_markdown(self.summary(lang=lang), **kwargs)

    def to_latex(self, lang: str = "en", **kwargs: Any) -> str:
        """Render the calibration summary table as LaTeX."""
        return df_to_latex(self.summary(lang=lang), **kwargs)

    def to_typst(self, lang: str = "en", **kwargs: Any) -> str:
        """Render the calibration summary table as Typst."""
        return df_to_typst(self.summary(lang=lang), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize calibrated economy parameters, audit metrics, and solver diagnostics."""
        return _json_safe({
            "countries": self.economy.countries,
            "sectors": self.economy.sectors,
            "parameters": {
                "beta": self.economy.beta,
                "delta": self.economy.delta,
                "capital_share": self.economy.capital_share,
                "risk_aversion": self.economy.risk_aversion,
                "trade_elasticity": self.economy.trade_elasticity,
                "capital_cost": self.economy.capital_cost,
                "labor_cost": self.economy.labor_cost,
                "sourcing_cost": self.economy.sourcing_cost,
                "productivity": self.economy.productivity,
                "sector_weights": self.economy.sector_weights,
                "home_shares": self.economy.home_shares,
                "labor_supply": self.economy.labor_supply,
            },
            "audit": {
                "source": self.audit.source,
                "country_groups": self.audit.country_groups,
                "sector_groups": self.audit.sector_groups,
                "expenditure_basis": self.audit.expenditure_basis,
                "pooled_capital_share": self.audit.pooled_capital_share,
                "empirical_cell_capital_shares": self.audit.empirical_cell_capital_shares,
                "world_va_total": self.audit.world_va_total,
                "labor_scale_factor": self.audit.labor_scale_factor,
                "empirical_va_shares": self.audit.empirical_va_shares,
                "steady_state_va_shares": self.audit.steady_state_va_shares,
                "va_share_autarky_gap": self.audit.va_share_autarky_gap,
                "stylized_assumptions": self.audit.stylized_assumptions,
                "warnings": self.audit.warnings,
            },
            "stability": dict(self.stability),
            "diagnostic": self.diagnostic.to_dict(),
        })


# ---------------------------------------------------------------------------
# Public ICIO -> ReallocationEconomy Calibration Function
# ---------------------------------------------------------------------------
def calibrate_reallocation_from_icio(
    icio: Any = None,
    *,
    country_groups: Mapping[str, Sequence[str]] | None = None,
    sector_groups: Mapping[str, Sequence[str]] | None = None,
    country_codes: Sequence[str] | None = None,
    sector_codes: Sequence[str] | None = None,
    fd_codes: Sequence[str] | None = None,
    expenditure_basis: Literal["final_demand", "consumption", "total_absorption"] = "final_demand",
    beta: float = 0.96,
    delta: float = 0.08,
    capital_share: float | None = None,
    risk_aversion: float = 2.0,
    trade_elasticity: float = 4.0,
    capital_cost: float = 4.0,
    labor_cost: float = 5.0,
    sourcing_cost: float = 0.01,
    productivity: Any | None = None,
    baseline_tariffs: Any | None = None,
    tolerance: float = 1e-8,
    max_evaluations: int = 200,
) -> ReallocationCalibrationResult:
    """Aggregate an ICIO table into a validated 2x2 :class:`ReallocationEconomy`.

    Static single-year ICIO accounts identify baseline expenditure shares,
    home-bias sourcing shares, value-added factor payments, and country labor
    income scale, whereas intertemporal preference, depreciation, elasticity,
    and quadratic adjustment-cost parameters are *not* identified from IO snapshots
    and are disclosed explicitly in ``result.audit.stylized_assumptions``.

    Note: ``sourcing_cost`` defaults to ``0.01`` (rather than the symmetric
    illustrative ``0.1``) so high empirical home-bias regions such as
    ``ROW`` (``home_shares ~ 0.98``) preserve a 16-root local stable saddle-path
    graph under :func:`~puremacro.trade.reallocation.reallocation_stability`.
    """
    if expenditure_basis not in _VALID_EXPENDITURE_BASES:
        raise ValueError(
            f"expenditure_basis must be one of {_VALID_EXPENDITURE_BASES}, got {expenditure_basis!r}"
        )

    io = _extract_io_arrays(icio, country_codes, sector_codes, fd_codes)
    coarse_countries, norm_c_groups, c_idx = _resolve_partition(
        io.country_codes, country_groups, kind="country"
    )
    coarse_sectors, norm_s_groups, s_idx = _resolve_partition(
        io.sector_codes, sector_groups, kind="sector"
    )
    agg = _aggregate_2x2(io, c_idx, s_idx)

    va_l = agg["VA_L"]
    va_k = agg["VA_K"]
    va = va_l + va_k
    if not np.isfinite(va).all() or (va <= 0).any() or (va_l <= 0).any() or (va_k <= 0).any():
        raise ValueError("Aggregated 2x2 labor and capital value added must be strictly positive")

    # Conserve total world VA and check aggregated cell totals match source sums.
    raw_va_total = float((io.labor_va + io.capital_va).sum())
    agg_va_total = float(va.sum())
    if abs(agg_va_total - raw_va_total) > 1e-8 * max(raw_va_total, 1.0):
        raise ValueError("Aggregation failed exact value-added conservation")

    cell_cap_shares = va_k / va
    if capital_share is None:
        pooled_cap = float(va_k.sum() / agg_va_total)
        cap_source = "pooled_icio"
    else:
        pooled_cap = real_scalar(capital_share, "capital_share")
        cap_source = "caller_override"

    f_tot = agg["F"].sum(axis=3)  # (src_c, src_s, dst_c)
    fine_f_tot = io.final_demand.sum(axis=3)  # (nc_fine, ns_fine, nc_fine)
    if "C" in io.fd_codes:
        c_fd_idx = io.fd_codes.index("C")
        f_cons: np.ndarray | None = agg["F"][:, :, :, c_fd_idx]
        fine_cons: np.ndarray | None = io.final_demand[:, :, :, c_fd_idx]
    else:
        if expenditure_basis == "consumption":
            raise ValueError(
                f"expenditure_basis='consumption' requires final-demand code 'C' in fd_codes={io.fd_codes}"
            )
        f_cons = None
        fine_cons = None

    if expenditure_basis == "final_demand":
        flow = f_tot
        fine_flow = fine_f_tot
    elif expenditure_basis == "consumption":
        assert f_cons is not None and fine_cons is not None
        flow = f_cons
        fine_flow = fine_cons
    else:
        flow = f_tot + agg["Z"].sum(axis=3)
        fine_flow = fine_f_tot + io.intermediate.sum(axis=3)

    if not np.isfinite(flow).all() or (flow <= 0).any():
        raise ValueError(
            f"Aggregated bilateral {expenditure_basis!r} flows must be strictly positive in all 2x2 cells"
        )

    exp_dst_sec = flow.sum(axis=0).T  # (dst_c, src_s)
    sector_weights = exp_dst_sec / exp_dst_sec.sum(axis=1, keepdims=True)

    home_shares = np.empty((2, 2), dtype=float)
    for c in range(2):
        for s in range(2):
            home_shares[c, s] = flow[c, s, c] / exp_dst_sec[c, s]
    if not ((home_shares > 0.0).all() and (home_shares < 1.0).all()):
        raise ValueError(f"Calibrated home_shares must lie strictly in (0, 1); got {home_shares.tolist()}")

    world_labor_va = float(va_l.sum())
    labor_scale_factor = world_labor_va / 2.0
    labor_supply = va_l.sum(axis=1) / labor_scale_factor

    prod_arr = np.ones((2, 2), dtype=float) if productivity is None else _check_matrix_2x2(productivity, "productivity")
    economy = ReallocationEconomy(
        countries=coarse_countries,
        sectors=coarse_sectors,
        beta=beta,
        delta=delta,
        capital_share=pooled_cap,
        risk_aversion=risk_aversion,
        trade_elasticity=trade_elasticity,
        capital_cost=capital_cost,
        labor_cost=labor_cost,
        sourcing_cost=sourcing_cost,
        productivity=prod_arr,
        sector_weights=sector_weights,
        home_shares=home_shares,
        labor_supply=labor_supply,
    )

    tau0 = np.zeros((2, 2), dtype=float) if baseline_tariffs is None else np.asarray(baseline_tariffs, dtype=float)
    steady = solve_reallocation_steady_state(
        economy, tariffs=tau0, tolerance=tolerance, max_evaluations=max_evaluations
    )
    ss_audit = audit_reallocation(steady, tolerance=tolerance)
    if not ss_audit["accepted"]:
        raise ValueError(f"Calibrated steady state failed reallocation audit (max_error={ss_audit['max_error']})")
    stability = reallocation_stability(steady)

    ss_va = steady.values["producer_prices"] * steady.values["output"]
    ss_va_shares = ss_va / ss_va.sum(axis=1, keepdims=True)
    emp_va_shares = va / va.sum(axis=1, keepdims=True)
    va_gap = float(np.max(np.abs(ss_va_shares - emp_va_shares)))

    warnings_list: list[str] = []
    if not stability["stable_graph"]:
        warnings_list.append(
            f"Calibrated steady state has {stability['stable_roots']}/16 stable roots (local stable graph check False); "
            "reduce sourcing_cost or inspect home_shares asymmetry before solving infinite-horizon transitions."
        )
    if float(np.max(np.abs(cell_cap_shares - pooled_cap))) > 0.05:
        warnings_list.append(
            "Empirical cell capital shares vary > 5pp across 2x2 cells; ReallocationEconomy pools a single scalar capital_share."
        )
    if va_gap > 0.01:
        warnings_list.append(
            f"Calibrated balanced-trade steady-state sectoral VA shares differ from raw ICIO VA shares by up to {va_gap:.3f} "
            "due to national financial autarky (zero trade deficit) and omission of intermediate input linkages."
        )

    stylized: dict[str, str] = {
        "beta": f"Discount factor={economy.beta} is not identified from a static IO table (stylized annual prior).",
        "delta": f"Capital depreciation rate={economy.delta} is not identified from a static IO table.",
        "risk_aversion": f"CRRA risk aversion={economy.risk_aversion} is not identified from a static IO table.",
        "trade_elasticity": f"Armington trade elasticity={economy.trade_elasticity} is not identified from a single IO cross section.",
        "adjustment_costs": (
            f"Quadratic adjustment costs (capital={economy.capital_cost}, labor={economy.labor_cost}, "
            f"sourcing={economy.sourcing_cost}) are not identified from static IO accounts."
        ),
        "productivity_normalization": (
            "Single-year nominal USD tables do not separate physical quantities from prices; "
            "unit TFP productivity=ones((2, 2)) is used unless caller supplies productivity."
            if productivity is None
            else "Caller-supplied productivity override applied."
        ),
        "factor_split": io.factor_split_note,
        "capital_share_pooling": (
            f"ReallocationEconomy requires a single global scalar capital_share ({economy.capital_share:.6g} from {cap_source})."
        ),
        "omitted_intermediates": (
            "ReallocationEconomy models value-added production only; ICIO intermediate matrix Z "
            f"(world sum={float(agg['Z'].sum()):.3f}) is omitted unless expenditure_basis='total_absorption'."
        ),
        "omitted_bilateral_tariffs": (
            "ICIO carries net product/production taxes (TLS/TFD) rather than bilateral MFN/preferential ad-valorem tariffs; "
            "baseline tariffs default to zero unless baseline_tariffs is passed."
        ),
        "national_financial_autarky": (
            "ReallocationEconomy imposes period-by-period balanced trade (net exports = 0), "
            f"producing a max regional VA share shift of {va_gap:.4f} relative to the unbalanced raw table."
        ),
    }

    audit_rec = ICIOCalibrationAudit(
        source=io.source,
        country_groups=norm_c_groups,
        sector_groups=norm_s_groups,
        source_countries=io.country_codes,
        source_sectors=io.sector_codes,
        expenditure_basis=expenditure_basis,
        pooled_capital_share=float(economy.capital_share),
        empirical_cell_capital_shares=cell_cap_shares,
        capital_share_source=cap_source,
        aggregated_va=va,
        aggregated_labor_va=va_l,
        aggregated_capital_va=va_k,
        aggregated_output=agg["output"],
        aggregated_final_demand=f_tot,
        aggregated_consumption=f_cons,
        aggregated_intermediate=agg["Z"],
        aggregated_tls=agg["TLS"],
        aggregated_tfd=agg["TFD"],
        fine_expenditure_flows=fine_flow,
        world_va_total=agg_va_total,
        world_labor_va_total=world_labor_va,
        labor_scale_factor=labor_scale_factor,
        empirical_va_shares=emp_va_shares,
        steady_state_va_shares=ss_va_shares,
        va_share_autarky_gap=va_gap,
        stylized_assumptions=stylized,
        warnings=tuple(warnings_list),
    )

    diag = make_solver_diagnostic_record(
        converged=bool(steady.converged),
        iterations=int(steady.diagnostics.get("nfev", 0)),
        stopping_reason=str(steady.diagnostics.get("message", "steady_state_solved")),
        solver_method="least_squares_trf",
        tolerance=tolerance,
        is_accurate=bool(ss_audit["accepted"]),
        max_residual=float(ss_audit["max_error"] or 0.0),
        residual_norm=float(ss_audit["max_error"] or 0.0),
        residual_type="reallocation_steady_state_audit",
        accuracy_tolerance=tolerance,
        residuals={str(k): float(v) for k, v in ss_audit["checks"].items()},
        is_supported=bool(stability["stable_graph"]),
        regime="stable_saddle_path" if stability["stable_graph"] else "unstable_root_count",
        warnings=warnings_list,
        boundary_metrics={
            "stable_roots": int(stability["stable_roots"]),
            "required_stable_roots": int(stability["required_stable_roots"]),
            "va_share_autarky_gap": va_gap,
        },
        evidence={"source": io.source, "expenditure_basis": expenditure_basis},
    )

    return ReallocationCalibrationResult(
        economy=economy,
        steady_state=steady,
        stability=stability,
        audit=audit_rec,
        diagnostic=diag,
        metadata={
            "expenditure_basis": expenditure_basis,
            "world_va_total": agg_va_total,
            "labor_scale_factor": labor_scale_factor,
        },
    )


def reallocation_tariffs_from_scenarios(
    scenario: TariffScenario | CompositeTariffScenario | Sequence[TariffScenario | CompositeTariffScenario],
    economy: ReallocationEconomy,
    *,
    horizon: int | None = None,
    calibration: ReallocationCalibrationResult | None = None,
) -> np.ndarray:
    """Convert :class:`TariffScenario` or :class:`CompositeTariffScenario` objects into a ``(T, 2, 2)`` path.

    Accepts scenarios defined either directly over coarse ``economy.countries`` and
    ``economy.sectors``, or over fine source ICIO codes when ``calibration`` is supplied
    (using baseline expenditure weights to aggregate bilateral import rates).
    """
    if isinstance(scenario, (TariffScenario, CompositeTariffScenario)):
        if horizon is None or horizon < 2:
            raise ValueError("horizon >= 2 is required when converting a single static tariff scenario")
        single = _single_scenario_matrix(scenario, economy, calibration)
        return np.broadcast_to(single[None, :, :], (int(horizon), 2, 2)).copy()

    seq = list(scenario)
    if len(seq) < 2:
        raise ValueError("scenario sequence must provide at least 2 dates (matching ReallocationTransitionResult)")
    if horizon is not None and int(horizon) != len(seq):
        raise ValueError(f"horizon={horizon} does not match len(scenario)={len(seq)}")
    out = np.empty((len(seq), 2, 2), dtype=float)
    for t, item in enumerate(seq):
        if not isinstance(item, (TariffScenario, CompositeTariffScenario)):
            raise TypeError(f"Expected TariffScenario or CompositeTariffScenario at date {t}, got {type(item).__name__}")
        out[t] = _single_scenario_matrix(item, economy, calibration)
    return out


def _iter_leaf_scenarios(comp: CompositeTariffScenario) -> list[TariffScenario]:
    leaves: list[TariffScenario] = []
    for item in comp.scenarios:
        if isinstance(item, CompositeTariffScenario):
            leaves.extend(_iter_leaf_scenarios(item))
        else:
            leaves.append(item)
    return leaves


def _single_scenario_matrix(
    scenario: TariffScenario | CompositeTariffScenario,
    economy: ReallocationEconomy,
    calibration: ReallocationCalibrationResult | None,
) -> np.ndarray:
    c_coarse = set(economy.countries)
    s_coarse = set(economy.sectors)
    if isinstance(scenario, TariffScenario):
        comp = CompositeTariffScenario((scenario,), name=scenario.name)
    else:
        comp = scenario

    leaves = _iter_leaf_scenarios(comp)
    referenced_countries = {s.importer for s in leaves}
    referenced_sectors: set[str] = set()
    array_shapes: set[tuple[int, ...]] = set()
    for s in leaves:
        if isinstance(s.tariffs, Mapping):
            for exp, sec in s.tariffs.keys():
                referenced_countries.add(str(exp))
                referenced_sectors.add(str(sec))
        else:
            array_shapes.add(np.asarray(s.tariffs).shape)

    matches_coarse_arrays = not array_shapes or array_shapes == {(2, 2)}
    if referenced_countries <= c_coarse and referenced_sectors <= s_coarse and matches_coarse_arrays:
        tensor = comp.to_rate_tensor(economy.countries, economy.sectors)  # (exp, imp, sec)
        mat = np.zeros((2, 2), dtype=float)
        for imp_i in range(2):
            exp_i = 1 - imp_i
            mat[imp_i, :] = tensor[exp_i, imp_i, :]
        return mat

    if calibration is None:
        raise ValueError(
            "Scenario uses disaggregate codes outside economy.countries/sectors; pass calibration= to aggregate"
        )
    src_c = calibration.audit.source_countries
    src_s = calibration.audit.source_sectors
    tensor_fine = comp.to_rate_tensor(src_c, src_s)  # (nc_fine, nc_fine, ns_fine)
    c_map = {m: idx for idx, g in enumerate(economy.countries) for m in calibration.audit.country_groups[g]}
    s_map = {m: idx for idx, g in enumerate(economy.sectors) for m in calibration.audit.sector_groups[g]}

    # Weight bilateral fine tariffs by baseline expenditure flows (src_c, src_s, dst_c)
    # retained under the calibration's expenditure_basis.
    weights_fine = calibration.audit.fine_expenditure_flows
    expected_shape = (len(src_c), len(src_s), len(src_c))
    if weights_fine.shape != expected_shape:
        raise ValueError(
            f"calibration.audit.fine_expenditure_flows must have shape {expected_shape}, got {weights_fine.shape}"
        )
    if not np.isfinite(weights_fine).all() or (weights_fine < 0.0).any():
        raise ValueError("calibration.audit.fine_expenditure_flows must be finite and nonnegative")

    mat = np.zeros((2, 2), dtype=float)
    weight_den = np.zeros((2, 2), dtype=float)
    for o_i, o_code in enumerate(src_c):
        go = c_map[o_code]
        for d_i, d_code in enumerate(src_c):
            gd = c_map[d_code]
            if go == gd:
                if np.any(tensor_fine[o_i, d_i, :] != 0.0):
                    raise ValueError(
                        f"ReallocationEconomy only models inter-regional tariffs; non-zero intra-bloc tariff "
                        f"between {o_code!r} and {d_code!r} inside {economy.countries[go]!r}"
                    )
                continue
            for s_i, s_code in enumerate(src_s):
                gs = s_map[s_code]
                w = float(weights_fine[o_i, s_i, d_i])
                mat[gd, gs] += w * float(tensor_fine[o_i, d_i, s_i])
                weight_den[gd, gs] += w
    if np.any(weight_den <= 0.0):
        raise ValueError(
            "Cannot aggregate fine bilateral tariffs: cross-region import expenditure weight is zero "
            "for at least one coarse (importer, sector) pair"
        )
    return mat / weight_den


# ---------------------------------------------------------------------------
# Household Group Specification & Helpers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HouseholdGroupSpec:
    """Income-source and expenditure-basket definition for one household group."""

    name: str
    country: str
    labor_weights: Any
    non_labor_weight: float | None = None
    capital_weight: float | None = None
    rebate_weight: float | None = None
    investment_weight: float | None = None
    expenditure_shares: Any | None = None
    rule: str | None = None
    expenditure_elasticities: Any | None = None
    supernumerary_share: float | None = None
    ces_elasticity: float | None = None

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        country = str(self.country).strip()
        if not name or not country:
            raise ValueError("HouseholdGroupSpec name and country must be non-empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "country", country)

        lw = _validate_nonneg_vector(self.labor_weights, f"groups[{name}].labor_weights", 2)
        object.__setattr__(self, "labor_weights", lw)

        nlw = self.non_labor_weight
        kw, rw, iw = self.capital_weight, self.rebate_weight, self.investment_weight
        if nlw is not None:
            nlw_f = _real_nonneg(nlw, f"groups[{name}].non_labor_weight")
            for label, explicit in (("capital_weight", kw), ("rebate_weight", rw), ("investment_weight", iw)):
                if explicit is not None and abs(float(explicit) - nlw_f) > 1e-12:
                    raise ValueError(
                        f"groups[{name}]: pass either non_labor_weight or explicit capital/rebate/investment weights"
                    )
            kw_f = rw_f = iw_f = nlw_f
        else:
            if kw is None or rw is None:
                raise ValueError(
                    f"groups[{name}]: specify either non_labor_weight or both capital_weight and rebate_weight"
                )
            kw_f = _real_nonneg(kw, f"groups[{name}].capital_weight")
            rw_f = _real_nonneg(rw, f"groups[{name}].rebate_weight")
            iw_f = kw_f if iw is None else _real_nonneg(iw, f"groups[{name}].investment_weight")

        object.__setattr__(self, "capital_weight", kw_f)
        object.__setattr__(self, "rebate_weight", rw_f)
        object.__setattr__(self, "investment_weight", iw_f)

        if self.expenditure_shares is not None:
            es = _validate_nonneg_vector(self.expenditure_shares, f"groups[{name}].expenditure_shares", 2)
            if abs(float(es.sum()) - 1.0) > 1e-10:
                raise ValueError(
                    f"groups[{name}].expenditure_shares must sum to 1.0 within 1e-10, got {float(es.sum()):.12g}"
                )
            object.__setattr__(self, "expenditure_shares", es)

        if self.rule is not None and self.rule not in _VALID_RULES:
            raise ValueError(f"groups[{name}].rule must be one of {_VALID_RULES}, got {self.rule!r}")

        if self.expenditure_elasticities is not None:
            ee = _validate_nonneg_vector(
                self.expenditure_elasticities, f"groups[{name}].expenditure_elasticities", 2
            )
            if (ee <= 0).any():
                raise ValueError(f"groups[{name}].expenditure_elasticities must be strictly positive")
            object.__setattr__(self, "expenditure_elasticities", ee)

        if self.supernumerary_share is not None:
            ss = real_scalar(self.supernumerary_share, f"groups[{name}].supernumerary_share")
            if not (0.0 < ss <= 1.0):
                raise ValueError(f"groups[{name}].supernumerary_share must lie in (0, 1]")
            object.__setattr__(self, "supernumerary_share", ss)

        if self.ces_elasticity is not None:
            ce = real_scalar(self.ces_elasticity, f"groups[{name}].ces_elasticity")
            if ce <= 0.0:
                raise ValueError(f"groups[{name}].ces_elasticity must be strictly positive")
            object.__setattr__(self, "ces_elasticity", ce)


def _real_nonneg(value: Any, name: str) -> float:
    val = real_scalar(value, name)
    if val < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {val}")
    return val


def _coerce_group_spec(item: HouseholdGroupSpec | Mapping[str, Any]) -> HouseholdGroupSpec:
    if isinstance(item, HouseholdGroupSpec):
        return item
    if isinstance(item, Mapping):
        return HouseholdGroupSpec(**dict(item))
    raise TypeError(f"Expected HouseholdGroupSpec or mapping, got {type(item).__name__}")


def make_sector_household_groups(
    economy: ReallocationEconomy,
    *,
    rebate_allocation: Literal["equal", "sector_weights"] = "equal",
    include_capital_owners: bool = False,
    basket_tilt: float = 0.0,
    rule: str | None = None,
) -> tuple[HouseholdGroupSpec, ...]:
    """Construct an exact partitioning set of sector-worker (and optional capitalist) groups.

    For each country ``c`` in ``economy.countries``:
    - With ``include_capital_owners=False``, creates two worker groups per country
      (``c_S0_Workers``, ``c_S1_Workers``) owning sector 0 and sector 1 labor
      respectively and splitting non-labor/capital/rebate flows equally (``0.5 / 0.5``)
      or according to ``economy.sector_weights[c]``.
    - With ``include_capital_owners=True``, creates two pure worker groups
      (zero capital/investment weight) and one ``c_Capital_Owners`` group
      (zero labor weight, ``capital_weight=investment_weight=1.0``), allocating
      rebates evenly across all three groups or evenly across workers when
      ``rebate_allocation="sector_weights"``.
    """
    tilt = real_scalar(basket_tilt, "basket_tilt")
    if rebate_allocation not in ("equal", "sector_weights"):
        raise ValueError("rebate_allocation must be 'equal' or 'sector_weights'")

    specs: list[HouseholdGroupSpec] = []
    s0, s1 = economy.sectors
    for c_idx, cname in enumerate(economy.countries):
        base_w = economy.sector_weights[c_idx].copy()
        w0 = np.clip(np.array([base_w[0] + tilt, base_w[1] - tilt]), 1e-6, 1.0 - 1e-6)
        w0 /= w0.sum()
        w1 = np.clip(np.array([base_w[0] - tilt, base_w[1] + tilt]), 1e-6, 1.0 - 1e-6)
        w1 /= w1.sum()
        if not include_capital_owners:
            splits = (0.5, 0.5) if rebate_allocation == "equal" else (float(base_w[0]), float(base_w[1]))
            specs.append(HouseholdGroupSpec(
                name=f"{cname}_{s0}_Workers",
                country=cname,
                labor_weights=(1.0, 0.0),
                non_labor_weight=splits[0],
                expenditure_shares=w0,
                rule=rule,
            ))
            specs.append(HouseholdGroupSpec(
                name=f"{cname}_{s1}_Workers",
                country=cname,
                labor_weights=(0.0, 1.0),
                non_labor_weight=splits[1],
                expenditure_shares=w1,
                rule=rule,
            ))
        else:
            reb = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0) if rebate_allocation == "equal" else (
                float(base_w[0]), float(base_w[1]), 0.0
            )
            specs.append(HouseholdGroupSpec(
                name=f"{cname}_{s0}_Workers",
                country=cname,
                labor_weights=(1.0, 0.0),
                capital_weight=0.0,
                investment_weight=0.0,
                rebate_weight=reb[0],
                expenditure_shares=w0,
                rule=rule,
            ))
            specs.append(HouseholdGroupSpec(
                name=f"{cname}_{s1}_Workers",
                country=cname,
                labor_weights=(0.0, 1.0),
                capital_weight=0.0,
                investment_weight=0.0,
                rebate_weight=reb[1],
                expenditure_shares=w1,
                rule=rule,
            ))
            specs.append(HouseholdGroupSpec(
                name=f"{cname}_Capital_Owners",
                country=cname,
                labor_weights=(0.0, 0.0),
                capital_weight=1.0,
                investment_weight=1.0,
                rebate_weight=reb[2],
                expenditure_shares=base_w,
                rule=rule,
            ))
    return tuple(specs)


# ---------------------------------------------------------------------------
# Transition Household-Group Incidence Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TransitionHouseholdIncidenceResult:
    """Transition welfare, cost-of-living, inequality, and conservation audit results."""

    economy: ReallocationEconomy
    transition: ReallocationTransitionResult
    groups: tuple[HouseholdGroupSpec, ...]
    group_names: tuple[str, ...]
    group_countries: tuple[str, ...]
    group_rules: tuple[str, ...]
    preferences: tuple[HouseholdPreferences, ...]
    baseline_budget: np.ndarray            # (G,)
    baseline_labor_income: np.ndarray      # (G,)
    baseline_capital_income: np.ndarray    # (G,)
    baseline_rebate_income: np.ndarray     # (G,)
    baseline_investment_outlay: np.ndarray # (G,)
    expenditure_shares: np.ndarray         # (G, 2)
    nominal_income: np.ndarray             # (T, G)
    real_income: np.ndarray                # (T, G) = nominal_income / cost_index
    equivalent_consumption: np.ndarray     # (T, G) = m0 + ev
    cost_index: np.ndarray                 # (T, G) Konus COLI e(p_t, u_0)/m_0
    price_index: np.ndarray                # (T, G) e(p_t, u_t)/e(1, u_t)
    ev: np.ndarray                         # (T, G)
    cv: np.ndarray                         # (T, G)
    ev_pct: np.ndarray                     # (T, G) = 100 * ev / m0
    cv_pct: np.ndarray                     # (T, G) = 100 * cv / m0
    labor_income: np.ndarray               # (T, G)
    capital_income: np.ndarray             # (T, G)
    rebate_income: np.ndarray              # (T, G)
    investment_outlay: np.ndarray          # (T, G)
    pv_ev_finite: np.ndarray               # (G,) sum_{t=0}^{T-1} beta^t ev[t, g]
    pv_ev_finite_pct: np.ndarray           # (G,) annuity % over finite horizon
    pv_ev: np.ndarray                      # (G,) with terminal steady-state continuation
    pv_ev_pct: np.ndarray                  # (G,) annuity % (1 - beta) * pv_ev / m0 * 100
    path_cev_pct: np.ndarray               # (G,) CRRA consumption equivalent %
    ev_pct_spread: np.ndarray              # (T, 2) max - min EV% per country
    ev_pct_std: np.ndarray                 # (T, 2) budget-weighted std of EV%
    cv_pct_spread: np.ndarray              # (T, 2)
    real_income_growth_spread: np.ndarray  # (T, 2) max(real/m0) - min(real/m0) in pp
    country_pv_ev_pct_spread: np.ndarray   # (2,)
    country_path_cev_pct_spread: np.ndarray  # (2,)
    audit_metrics: dict[str, Any]
    diagnostic: SolverDiagnosticRecord
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        arr_fields = (
            "baseline_budget",
            "baseline_labor_income",
            "baseline_capital_income",
            "baseline_rebate_income",
            "baseline_investment_outlay",
            "expenditure_shares",
            "nominal_income",
            "real_income",
            "equivalent_consumption",
            "cost_index",
            "price_index",
            "ev",
            "cv",
            "ev_pct",
            "cv_pct",
            "labor_income",
            "capital_income",
            "rebate_income",
            "investment_outlay",
            "pv_ev_finite",
            "pv_ev_finite_pct",
            "pv_ev",
            "pv_ev_pct",
            "path_cev_pct",
            "ev_pct_spread",
            "ev_pct_std",
            "cv_pct_spread",
            "real_income_growth_spread",
            "country_pv_ev_pct_spread",
            "country_path_cev_pct_spread",
        )
        for fname in arr_fields:
            object.__setattr__(self, fname, readonly_array(np.array(getattr(self, fname), dtype=float, copy=True)))
        object.__setattr__(self, "audit_metrics", _freeze_metadata(self.audit_metrics))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def summary(self, lang: str = "en") -> pd.DataFrame:
        """Return group-level transition incidence and lifetime/PV welfare table."""
        lcode = _validate_lang(lang)
        rows: list[dict[str, Any]] = []
        for g_idx, gname in enumerate(self.group_names):
            cname = self.group_countries[g_idx]
            m0 = float(self.baseline_budget[g_idx])
            rows.append({
                "Group" if lcode == "en" else "Grupo": gname,
                "Country" if lcode == "en" else "País": cname,
                "Rule" if lcode == "en" else "Regla": self.group_rules[g_idx],
                "Base_Budget" if lcode == "en" else "Presupuesto_Base": m0,
                "Impact_Cost_Index" if lcode == "en" else "Índice_Costo_Impacto": float(self.cost_index[0, g_idx]),
                "Impact_Real_Income_%Chg" if lcode == "en" else "Ingreso_Real_Impacto_%": float(
                    100.0 * (self.real_income[0, g_idx] / m0 - 1.0)
                ),
                "Impact_EV_%" if lcode == "en" else "VE_Impacto_%": float(self.ev_pct[0, g_idx]),
                "Impact_CV_%" if lcode == "en" else "VC_Impacto_%": float(self.cv_pct[0, g_idx]),
                "Terminal_EV_%" if lcode == "en" else "VE_Terminal_%": float(self.ev_pct[-1, g_idx]),
                "PV_EV_%" if lcode == "en" else "VP_VE_%": float(self.pv_ev_pct[g_idx]),
                "Path_CEV_%" if lcode == "en" else "VEC_Senda_%": float(self.path_cev_pct[g_idx]),
            })
        return pd.DataFrame(rows)

    def to_dataframe(self, lang: str = "en") -> pd.DataFrame:
        """Alias for :meth:`summary`."""
        return self.summary(lang=lang)

    def to_frame(self, lang: str = "en") -> pd.DataFrame:
        """Alias for :meth:`summary`."""
        return self.summary(lang=lang)

    def path_dataframe(self) -> pd.DataFrame:
        """Return full ``(date, group)`` transition panel across dates ``0 .. T-1``."""
        T, G = self.ev.shape
        records: list[dict[str, Any]] = []
        for t in range(T):
            for g in range(G):
                m0 = float(self.baseline_budget[g])
                records.append({
                    "date": t,
                    "group": self.group_names[g],
                    "country": self.group_countries[g],
                    "nominal_income": float(self.nominal_income[t, g]),
                    "real_income": float(self.real_income[t, g]),
                    "real_income_pct_chg": float(100.0 * (self.real_income[t, g] / m0 - 1.0)),
                    "cost_index": float(self.cost_index[t, g]),
                    "price_index": float(self.price_index[t, g]),
                    "ev": float(self.ev[t, g]),
                    "cv": float(self.cv[t, g]),
                    "ev_pct": float(self.ev_pct[t, g]),
                    "cv_pct": float(self.cv_pct[t, g]),
                    "labor_income": float(self.labor_income[t, g]),
                    "capital_income": float(self.capital_income[t, g]),
                    "rebate_income": float(self.rebate_income[t, g]),
                    "investment_outlay": float(self.investment_outlay[t, g]),
                })
        return pd.DataFrame(records)

    def inequality_dataframe(self) -> pd.DataFrame:
        """Return date-by-date cross-group inequality gap metrics per country."""
        T = self.ev.shape[0]
        records: list[dict[str, Any]] = []
        for t in range(T):
            for c_idx, cname in enumerate(self.economy.countries):
                g_indices = [i for i, gc in enumerate(self.group_countries) if gc == cname]
                evs_t = self.ev_pct[t, g_indices]
                top_local = int(np.argmax(evs_t))
                bot_local = int(np.argmin(evs_t))
                records.append({
                    "date": t,
                    "country": cname,
                    "ev_pct_spread_pp": float(self.ev_pct_spread[t, c_idx]),
                    "ev_pct_std_pp": float(self.ev_pct_std[t, c_idx]),
                    "cv_pct_spread_pp": float(self.cv_pct_spread[t, c_idx]),
                    "real_income_growth_spread_pp": float(self.real_income_growth_spread[t, c_idx]),
                    "top_group": self.group_names[g_indices[top_local]],
                    "bottom_group": self.group_names[g_indices[bot_local]],
                })
        return pd.DataFrame(records)

    def to_markdown(self, lang: str = "en", **kwargs: Any) -> str:
        """Render the household-group incidence summary table as Markdown."""
        return df_to_markdown(self.summary(lang=lang), **kwargs)

    def to_latex(self, lang: str = "en", **kwargs: Any) -> str:
        """Render the household-group incidence summary table as LaTeX."""
        return df_to_latex(self.summary(lang=lang), **kwargs)

    def to_typst(self, lang: str = "en", **kwargs: Any) -> str:
        """Render the household-group incidence summary table as Typst."""
        return df_to_typst(self.summary(lang=lang), **kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Serialize incidence paths, PV/CEV summaries, inequality gaps, and audits."""
        return _json_safe({
            "countries": self.economy.countries,
            "sectors": self.economy.sectors,
            "group_names": self.group_names,
            "group_countries": self.group_countries,
            "group_rules": self.group_rules,
            "baseline_budget": self.baseline_budget,
            "expenditure_shares": self.expenditure_shares,
            "nominal_income": self.nominal_income,
            "real_income": self.real_income,
            "equivalent_consumption": self.equivalent_consumption,
            "cost_index": self.cost_index,
            "price_index": self.price_index,
            "ev": self.ev,
            "cv": self.cv,
            "ev_pct": self.ev_pct,
            "cv_pct": self.cv_pct,
            "pv_ev_finite": self.pv_ev_finite,
            "pv_ev_finite_pct": self.pv_ev_finite_pct,
            "pv_ev": self.pv_ev,
            "pv_ev_pct": self.pv_ev_pct,
            "path_cev_pct": self.path_cev_pct,
            "ev_pct_spread": self.ev_pct_spread,
            "ev_pct_std": self.ev_pct_std,
            "cv_pct_spread": self.cv_pct_spread,
            "real_income_growth_spread": self.real_income_growth_spread,
            "country_pv_ev_pct_spread": self.country_pv_ev_pct_spread,
            "country_path_cev_pct_spread": self.country_path_cev_pct_spread,
            "audit_metrics": dict(self.audit_metrics),
            "diagnostic": self.diagnostic.to_dict(),
        })


# ---------------------------------------------------------------------------
# Transition Validation, Household Calibration & Incidence Evaluation
# ---------------------------------------------------------------------------
def _validate_transition_ready(
    transition: Any,
    audit_tolerance: float,
) -> tuple[ReallocationTransitionResult, dict[str, Any]]:
    if not isinstance(transition, ReallocationTransitionResult):
        raise TransitionError(
            f"transition must be a ReallocationTransitionResult, got {type(transition).__name__}"
        )
    if not transition.converged:
        raise TransitionError(
            "Refusing household incidence evaluation on an unconverged ReallocationTransitionResult "
            f"(status={transition.diagnostics.get('message', 'not accepted')!r})"
        )
    if not transition.initial.converged or not transition.terminal.converged:
        raise TransitionError("Transition initial and terminal steady states must be converged")
    audit = audit_reallocation(transition, tolerance=audit_tolerance)
    if not audit["accepted"]:
        raise TransitionError(
            "Refusing household incidence evaluation because ReallocationTransitionResult failed "
            f"reallocation accounting audit (max_error={audit['max_error']})"
        )
    return transition, audit


def _validate_and_partition_groups(
    groups: Sequence[HouseholdGroupSpec | Mapping[str, Any]],
    economy: ReallocationEconomy,
    weight_tolerance: float,
) -> tuple[tuple[HouseholdGroupSpec, ...], dict[str, list[int]]]:
    specs = tuple(_coerce_group_spec(g) for g in groups)
    if not specs:
        raise ValueError("groups cannot be empty")
    seen_names: set[str] = set()
    by_country: dict[str, list[int]] = {c: [] for c in economy.countries}
    for idx, spec in enumerate(specs):
        if spec.name in seen_names:
            raise ValueError(f"Duplicate household group name {spec.name!r}")
        seen_names.add(spec.name)
        if spec.country not in by_country:
            raise ValueError(
                f"Household group {spec.name!r} references unknown country {spec.country!r}; "
                f"expected one of {economy.countries}"
            )
        by_country[spec.country].append(idx)

    for cname, idxs in by_country.items():
        if not idxs:
            raise ValueError(f"Country {cname!r} has no household groups; both countries must be covered")
        lw_sum = sum(specs[i].labor_weights for i in idxs)
        if float(np.max(np.abs(lw_sum - 1.0))) > weight_tolerance:
            raise ValueError(
                f"Sector labor_weights for country {cname!r} must sum to 1.0 per sector, "
                f"got {np.asarray(lw_sum).tolist()}"
            )
        for field_name in ("capital_weight", "rebate_weight", "investment_weight"):
            total = sum(float(getattr(specs[i], field_name)) for i in idxs)
            if abs(total - 1.0) > weight_tolerance:
                raise ValueError(
                    f"{field_name} for country {cname!r} must sum to 1.0, got {total:.12g}"
                )
    return specs, by_country


def _compute_group_budgets_at(
    values: Mapping[str, np.ndarray],
    specs: tuple[HouseholdGroupSpec, ...],
    c_to_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute nominal income components for one date or steady-state mapping."""
    w_bill = values["wages"] * values["labor"]            # (..., 2, 2)
    r_tot = np.sum(values["rentals"] * values["capital"], axis=-1)  # (..., 2)
    reb = values["rebates"]                               # (..., 2)
    inv_adj = values["price_index"] * (values["gross_absorption"] - values["consumption"])

    lead_shape = r_tot.shape[:-1]
    G = len(specs)
    labor_inc = np.empty(lead_shape + (G,), dtype=float)
    cap_inc = np.empty(lead_shape + (G,), dtype=float)
    reb_inc = np.empty(lead_shape + (G,), dtype=float)
    inv_out = np.empty(lead_shape + (G,), dtype=float)

    for g_idx, sp in enumerate(specs):
        c = c_to_idx[sp.country]
        lw = np.asarray(sp.labor_weights, dtype=float)
        labor_inc[..., g_idx] = (
            lw[0] * w_bill[..., c, 0] + lw[1] * w_bill[..., c, 1]
        )
        assert sp.capital_weight is not None
        assert sp.rebate_weight is not None
        assert sp.investment_weight is not None
        cap_inc[..., g_idx] = float(sp.capital_weight) * r_tot[..., c]
        reb_inc[..., g_idx] = float(sp.rebate_weight) * reb[..., c]
        inv_out[..., g_idx] = float(sp.investment_weight) * inv_adj[..., c]

    budget = labor_inc + cap_inc + reb_inc - inv_out
    return budget, labor_inc, cap_inc, reb_inc, inv_out


def _calibrate_group_preferences(
    specs: tuple[HouseholdGroupSpec, ...],
    economy: ReallocationEconomy,
    c_to_idx: dict[str, int],
    m0: np.ndarray,
    default_rule: str,
    default_ces_elasticity: float,
    default_supernumerary_share: float | None,
    default_expenditure_elasticities: Any | None,
) -> tuple[np.ndarray, tuple[str, ...], list[tuple[list[int], HouseholdPreferences]]]:
    if default_rule not in _VALID_RULES:
        raise ValueError(f"rule must be one of {_VALID_RULES}, got {default_rule!r}")
    G = len(specs)
    shares = np.empty((G, 2), dtype=float)
    group_rules: list[str] = []
    buckets: dict[tuple[Any, ...], list[int]] = {}

    for g_idx, sp in enumerate(specs):
        c = c_to_idx[sp.country]
        sh = economy.sector_weights[c] if sp.expenditure_shares is None else sp.expenditure_shares
        shares[g_idx] = np.asarray(sh, dtype=float)
        r = default_rule if sp.rule is None else sp.rule
        group_rules.append(r)

        if r == "ces":
            ce = float(default_ces_elasticity if sp.ces_elasticity is None else sp.ces_elasticity)
            key: tuple[Any, ...] = ("ces", ce, None)
        elif r == "stone_geary":
            sup = default_supernumerary_share if sp.supernumerary_share is None else sp.supernumerary_share
            # Keep heterogeneous per-group expenditure_elasticities in separate buckets unless uniform/None.
            ee_key = (
                tuple(np.asarray(sp.expenditure_elasticities, dtype=float).tolist())
                if sp.expenditure_elasticities is not None
                else ("__default__",)
            )
            key = ("stone_geary", sup, ee_key)
        else:
            key = (r, None, None)
        buckets.setdefault(key, []).append(g_idx)

    calibrated_buckets: list[tuple[list[int], HouseholdPreferences]] = []
    for (r, param, ee_key), idxs in buckets.items():
        x0_bucket = np.column_stack([shares[g] * m0[g] for g in idxs])  # (2, len(idxs))
        names_bucket = tuple(specs[g].name for g in idxs)
        kw: dict[str, Any] = {
            "rule": r,
            "sector_codes": economy.sectors,
            "country_codes": names_bucket,
            "repair_adding_up": False,
        }
        if r == "ces":
            kw["ces_elasticity"] = float(param)
        elif r == "stone_geary":
            if param is not None:
                kw["supernumerary_share"] = float(param)
            if ee_key != ("__default__",):
                kw["expenditure_elasticities"] = np.column_stack(
                    [np.asarray(specs[g].expenditure_elasticities, dtype=float) for g in idxs]
                )
            elif default_expenditure_elasticities is not None:
                kw["expenditure_elasticities"] = default_expenditure_elasticities
        pref = calibrate_household(x0_bucket, **kw)
        calibrated_buckets.append((idxs, pref))

    return shares, tuple(group_rules), calibrated_buckets


def _evaluate_welfare_for_bucket(
    pref: HouseholdPreferences,
    idxs: list[int],
    rel_prices: np.ndarray,   # (..., 2[country], 2[sector])
    budget: np.ndarray,       # (..., G)
    c_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate benchmark-normalized HouseholdWelfareResult along path or single point."""
    lead_shape = budget.shape[:-1]
    n_b = len(idxs)
    out_shape = lead_shape + (n_b,)
    ev = np.empty(out_shape, dtype=float)
    cv = np.empty(out_shape, dtype=float)
    ev_pct = np.empty(out_shape, dtype=float)
    cv_pct = np.empty(out_shape, dtype=float)
    cost_idx = np.empty(out_shape, dtype=float)
    price_idx = np.empty(out_shape, dtype=float)

    if len(lead_shape) == 0:
        p_mat = np.column_stack([rel_prices[c_indices[g], :] for g in idxs])
        m_vec = np.array([budget[g] for g in idxs], dtype=float)
        res = compute_household_welfare(pref, p_mat, m_vec)
        return (
            np.asarray(res.ev, dtype=float),
            np.asarray(res.cv, dtype=float),
            np.asarray(res.ev_pct_consumption, dtype=float),
            np.asarray(res.cv_pct_consumption, dtype=float),
            np.asarray(res.cost_index, dtype=float),
            np.asarray(res.price_index, dtype=float),
        )

    T = lead_shape[0]
    for t in range(T):
        p_mat = np.column_stack([rel_prices[t, c_indices[g], :] for g in idxs])
        m_vec = np.array([budget[t, g] for g in idxs], dtype=float)
        res = compute_household_welfare(pref, p_mat, m_vec)
        ev[t] = res.ev
        cv[t] = res.cv
        ev_pct[t] = res.ev_pct_consumption
        cv_pct[t] = res.cv_pct_consumption
        cost_idx[t] = res.cost_index
        price_idx[t] = res.price_index
    return ev, cv, ev_pct, cv_pct, cost_idx, price_idx


def evaluate_transition_household_incidence(
    transition: ReallocationTransitionResult,
    groups: Sequence[HouseholdGroupSpec | Mapping[str, Any]],
    *,
    rule: Literal["fixed_baskets", "cobb_douglas", "stone_geary", "ces"] = "cobb_douglas",
    ces_elasticity: float = 1.5,
    supernumerary_share: float | None = 0.5,
    expenditure_elasticities: Any | None = None,
    audit_tolerance: float = 1e-8,
    weight_tolerance: float = 1e-10,
    conservation_tolerance: float = 1e-8,
) -> TransitionHouseholdIncidenceResult:
    """Evaluate dynamic household-group incidence along a converged reallocation path.

    Takes a converged, audited :class:`ReallocationTransitionResult` and an exact
    country-by-country household-group decomposition (:class:`HouseholdGroupSpec`),
    calibrates benchmark-normalized demand preferences via :mod:`puremacro.trade.household`,
    and returns period-by-period cost-of-living indices, real incomes, Hicksian ``EV_t``
    and ``CV_t``, present-value path EV, CRRA path CEV, cross-group inequality spreads,
    conservation residuals, zero-shock invariants, and a :class:`SolverDiagnosticRecord`.
    """
    tr, tr_audit = _validate_transition_ready(transition, audit_tolerance)
    economy = tr.economy
    c_to_idx = {c: i for i, c in enumerate(economy.countries)}
    specs, by_country = _validate_and_partition_groups(groups, economy, weight_tolerance)
    G = len(specs)
    T = len(tr.tariffs)
    c_indices = [c_to_idx[sp.country] for sp in specs]

    # Initial steady-state, transition path, and terminal steady-state budgets.
    m0, l0, k0, r0, inv0 = _compute_group_budgets_at(tr.initial.values, specs, c_to_idx)
    if not np.isfinite(m0).all() or (m0 <= 0).any():
        raise TransitionError("Every household group must have strictly positive baseline consumption budget")

    m_path, l_path, k_path, r_path, inv_path = _compute_group_budgets_at(tr.paths, specs, c_to_idx)
    if not np.isfinite(m_path).all() or (m_path <= 0).any():
        raise TransitionError("Group consumption budget became nonpositive along the transition path")

    m_term, _, _, _, _ = _compute_group_budgets_at(tr.terminal.values, specs, c_to_idx)
    if not np.isfinite(m_term).all() or (m_term <= 0).any():
        raise TransitionError("Group consumption budget is nonpositive in terminal steady state")

    # Calibrate preferences on baseline benchmark (p0 = ones, expenditure0 = shares * m0).
    shares, group_rules, buckets = _calibrate_group_preferences(
        specs,
        economy,
        c_to_idx,
        m0,
        default_rule=rule,
        default_ces_elasticity=ces_elasticity,
        default_supernumerary_share=supernumerary_share,
        default_expenditure_elasticities=expenditure_elasticities,
    )

    ps0 = tr.initial.values["sector_prices"]  # (2, 2)
    rel_p_path = tr.paths["sector_prices"] / ps0[None, :, :]  # (T, 2, 2)
    rel_p_term = tr.terminal.values["sector_prices"] / ps0     # (2, 2)

    ev = np.empty((T, G), dtype=float)
    cv = np.empty((T, G), dtype=float)
    ev_pct = np.empty((T, G), dtype=float)
    cv_pct = np.empty((T, G), dtype=float)
    cost_index = np.empty((T, G), dtype=float)
    price_index = np.empty((T, G), dtype=float)
    ev_term = np.empty(G, dtype=float)

    for idxs, pref in buckets:
        b_ev, b_cv, b_evp, b_cvp, b_ci, b_pi = _evaluate_welfare_for_bucket(
            pref, idxs, rel_p_path, m_path, c_indices
        )
        t_ev, _, _, _, _, _ = _evaluate_welfare_for_bucket(
            pref, idxs, rel_p_term, m_term, c_indices
        )
        for pos, g in enumerate(idxs):
            ev[:, g] = b_ev[:, pos]
            cv[:, g] = b_cv[:, pos]
            ev_pct[:, g] = b_evp[:, pos]
            cv_pct[:, g] = b_cvp[:, pos]
            cost_index[:, g] = b_ci[:, pos]
            price_index[:, g] = b_pi[:, pos]
            ev_term[g] = t_ev[pos]

    real_income = m_path / cost_index
    equiv_cons = m0[None, :] + ev
    equiv_cons_term = m0 + ev_term

    # Discounted present-value path EV & intertemporal CRRA consumption equivalent.
    discounts = economy.beta ** np.arange(T, dtype=float)
    pv_ev_finite = discounts @ ev
    pv_ev_finite_pct = 100.0 * pv_ev_finite / (m0 * float(discounts.sum()))
    pv_ev = pv_ev_finite + (economy.beta ** T / (1.0 - economy.beta)) * ev_term
    pv_ev_pct = 100.0 * (1.0 - economy.beta) * pv_ev / m0

    if np.all(equiv_cons > 0) and np.all(equiv_cons_term > 0):
        crra_res = consumption_equivalent_welfare(
            equiv_cons,
            baseline_consumption=m0,
            beta=economy.beta,
            risk_aversion=economy.risk_aversion,
            terminal_consumption=equiv_cons_term,
        )
        path_cev_pct = np.asarray(crra_res.consumption_equivalent_pct, dtype=float)
    else:
        raise TransitionError("Equivalent consumption dropped outside positive CRRA welfare domain")

    # Cross-group inequality spreads per country along t = 0 .. T-1.
    ev_pct_spread = np.zeros((T, 2), dtype=float)
    ev_pct_std = np.zeros((T, 2), dtype=float)
    cv_pct_spread = np.zeros((T, 2), dtype=float)
    real_inc_spread = np.zeros((T, 2), dtype=float)
    country_pv_spread = np.zeros(2, dtype=float)
    country_cev_spread = np.zeros(2, dtype=float)

    for c_idx, cname in enumerate(economy.countries):
        g_idxs = by_country[cname]
        if len(g_idxs) >= 2:
            ev_sub = ev_pct[:, g_idxs]
            cv_sub = cv_pct[:, g_idxs]
            w_sub = m0[g_idxs] / float(m0[g_idxs].sum())
            mean_ev = (ev_sub @ w_sub)[:, None]
            ev_pct_spread[:, c_idx] = ev_sub.max(axis=1) - ev_sub.min(axis=1)
            ev_pct_std[:, c_idx] = np.sqrt(((ev_sub - mean_ev) ** 2) @ w_sub)
            cv_pct_spread[:, c_idx] = cv_sub.max(axis=1) - cv_sub.min(axis=1)
            rg_sub = 100.0 * (real_income[:, g_idxs] / m0[None, g_idxs] - 1.0)
            real_inc_spread[:, c_idx] = rg_sub.max(axis=1) - rg_sub.min(axis=1)
            country_pv_spread[c_idx] = float(pv_ev_pct[g_idxs].max() - pv_ev_pct[g_idxs].min())
            country_cev_spread[c_idx] = float(path_cev_pct[g_idxs].max() - path_cev_pct[g_idxs].min())

    # Conservation & consistency audits along transition path t = 0 .. T-1.
    nat_cons_nom = tr.paths["price_index"] * tr.paths["consumption"]       # (T, 2)
    nat_abs_nom = tr.paths["price_index"] * tr.paths["gross_absorption"]   # (T, 2)
    nat_wage_bill = tr.paths["wages"] * tr.paths["labor"]                  # (T, 2, 2)

    max_cons_gap = 0.0
    max_nat_inc_gap = 0.0
    max_lab_gap = 0.0
    for c_idx, cname in enumerate(economy.countries):
        g_idxs = by_country[cname]
        grp_cons_sum = m_path[:, g_idxs].sum(axis=1)
        grp_gross_inc_sum = (l_path[:, g_idxs] + k_path[:, g_idxs] + r_path[:, g_idxs]).sum(axis=1)
        grp_lab_by_sec = np.zeros((T, 2), dtype=float)
        for g in g_idxs:
            lw = np.asarray(specs[g].labor_weights, dtype=float)
            grp_lab_by_sec[:, 0] += lw[0] * nat_wage_bill[:, c_idx, 0]
            grp_lab_by_sec[:, 1] += lw[1] * nat_wage_bill[:, c_idx, 1]
        max_cons_gap = max(max_cons_gap, float(np.max(np.abs(grp_cons_sum - nat_cons_nom[:, c_idx]))))
        max_nat_inc_gap = max(max_nat_inc_gap, float(np.max(np.abs(grp_gross_inc_sum - nat_abs_nom[:, c_idx]))))
        max_lab_gap = max(max_lab_gap, float(np.max(np.abs(grp_lab_by_sec - nat_wage_bill[:, c_idx]))))

    max_share_gap = float(np.max(np.abs(shares.sum(axis=1) - 1.0)))
    zero_shock = bool(np.max(np.abs(tr.tariffs)) == 0.0)
    max_zero_ev = float(np.max(np.abs(ev))) if zero_shock else 0.0
    max_zero_cv = float(np.max(np.abs(cv))) if zero_shock else 0.0
    max_zero_ci = float(np.max(np.abs(cost_index - 1.0))) if zero_shock else 0.0
    max_cons_residual = max(max_cons_gap, max_nat_inc_gap, max_lab_gap, max_share_gap)

    if max_cons_residual > conservation_tolerance:
        raise TransitionError(
            f"Household conservation audit failed: max residual {max_cons_residual:.3e} > {conservation_tolerance:.3e}"
        )
    if zero_shock and max(max_zero_ev, max_zero_cv, max_zero_ci) > 1e-12:
        raise TransitionError(
            "Zero-tariff-shock identity violated: "
            f"max|EV|={max_zero_ev:.3e}, max|CV|={max_zero_cv:.3e}, max|cost_index-1|={max_zero_ci:.3e}"
        )

    audit_metrics: dict[str, Any] = {
        "max_country_consumption_gap": max_cons_gap,
        "max_national_income_gap": max_nat_inc_gap,
        "max_sector_labor_gap": max_lab_gap,
        "max_expenditure_share_gap": max_share_gap,
        "max_conservation_residual": max_cons_residual,
        "zero_shock_detected": zero_shock,
        "zero_shock_max_abs_ev": max_zero_ev if zero_shock else None,
        "zero_shock_max_abs_cv": max_zero_cv if zero_shock else None,
        "zero_shock_max_cost_index_error": max_zero_ci if zero_shock else None,
        "transition_audit_max_error": float(tr_audit["max_error"] or 0.0),
    }

    overall_res = max(max_cons_residual, float(tr_audit["max_error"] or 0.0))
    warnings_list: list[str] = []
    if not tr.diagnostics.get("stable_terminal_required", True):
        warnings_list.append("Underlying transition solved with require_stable_terminal=False (finite boundary experiment).")

    diag = make_solver_diagnostic_record(
        converged=bool(tr.converged),
        iterations=int(tr.diagnostics.get("nfev", 0)),
        stopping_reason="household_incidence_reconciled",
        solver_method="reallocation_transition_plus_household_ev_cv",
        tolerance=conservation_tolerance,
        is_accurate=bool(max_cons_residual <= conservation_tolerance and tr_audit["accepted"]),
        max_residual=overall_res,
        residual_norm=overall_res,
        residual_type="budget_and_factor_conservation",
        accuracy_tolerance=conservation_tolerance,
        residuals={
            "max_country_consumption_gap": max_cons_gap,
            "max_national_income_gap": max_nat_inc_gap,
            "max_sector_labor_gap": max_lab_gap,
            "max_expenditure_share_gap": max_share_gap,
            "transition_audit_max_error": float(tr_audit["max_error"] or 0.0),
        },
        is_supported=True,
        regime="zero_shock_identity" if zero_shock else "policy_transition",
        warnings=warnings_list,
        boundary_metrics={
            "horizon": T,
            "n_groups": G,
            "terminal_gap": float(tr.diagnostics.get("terminal_gap", 0.0)),
        },
        evidence={
            "rules": list(dict.fromkeys(group_rules)),
            "zero_shock_detected": zero_shock,
        },
    )

    return TransitionHouseholdIncidenceResult(
        economy=economy,
        transition=tr,
        groups=specs,
        group_names=tuple(sp.name for sp in specs),
        group_countries=tuple(sp.country for sp in specs),
        group_rules=group_rules,
        preferences=tuple(p for _, p in buckets),
        baseline_budget=m0,
        baseline_labor_income=l0,
        baseline_capital_income=k0,
        baseline_rebate_income=r0,
        baseline_investment_outlay=inv0,
        expenditure_shares=shares,
        nominal_income=m_path,
        real_income=real_income,
        equivalent_consumption=equiv_cons,
        cost_index=cost_index,
        price_index=price_index,
        ev=ev,
        cv=cv,
        ev_pct=ev_pct,
        cv_pct=cv_pct,
        labor_income=l_path,
        capital_income=k_path,
        rebate_income=r_path,
        investment_outlay=inv_path,
        pv_ev_finite=pv_ev_finite,
        pv_ev_finite_pct=pv_ev_finite_pct,
        pv_ev=pv_ev,
        pv_ev_pct=pv_ev_pct,
        path_cev_pct=path_cev_pct,
        ev_pct_spread=ev_pct_spread,
        ev_pct_std=ev_pct_std,
        cv_pct_spread=cv_pct_spread,
        real_income_growth_spread=real_inc_spread,
        country_pv_ev_pct_spread=country_pv_spread,
        country_path_cev_pct_spread=country_cev_spread,
        audit_metrics=audit_metrics,
        diagnostic=diag,
        metadata={
            "rule_default": rule,
            "ces_elasticity_default": ces_elasticity,
            "supernumerary_share_default": supernumerary_share,
        },
    )
