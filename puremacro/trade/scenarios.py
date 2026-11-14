"""Explicit tariff scenarios, multi-importer composition, and batch comparison.

A :class:`TariffScenario` names one importing country and the ad valorem rates it
levies by exporter and sector. :class:`CompositeTariffScenario` and
:func:`combine_scenarios` compose several scenarios (for example simultaneous
retaliation or carve-outs) with explicit conflict-resolution rules.
:func:`build_tariff_matrices` turns either scenario type into the bilateral
multiplier tensors of a calibration, :func:`run_scenario` solves one scenario,
:func:`run_scenario_batch` solves several from a common baseline (warm-started)
with Geary-Khamis real GDP, and :func:`summarize_scenario_batch` pairs those
results with Hicksian equivalent/compensating variation and robust-policy
ranking via :class:`ScenarioComparisonResult`.

There is no registry of named scenarios: every scenario states its tariffs.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
import math
from numbers import Real
from typing import TYPE_CHECKING, Any, Literal, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from puremacro.reports import df_to_latex, df_to_markdown, df_to_typst
from puremacro.trade._results import (
    GearyKhamisResult,
    ScenarioBatchResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from puremacro.trade.geary_khamis import compute_geary_khamis
from puremacro.trade.solver import build_initial_guess, solve_trade_equilibrium

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from puremacro.study.robust import RobustPolicyComparison

NAMED_SCENARIOS_REMOVED = "named scenarios were removed; build a TariffScenario"

ALL_SECTORS = "*"
"""Sector wildcard in :attr:`TariffScenario.tariffs` keys: every sector of the exporter."""

ConflictMode = Literal["last_wins", "first_wins", "error", "max", "sum"]
VALID_CONFLICT_MODES: tuple[ConflictMode, ...] = (
    "last_wins",
    "first_wins",
    "error",
    "max",
    "sum",
)

_WELFARE_METRICS = frozenset({
    "ev",
    "cv",
    "ev_pct_consumption",
    "ev_pct_gdp",
    "price_effect",
    "factor_income_effect",
})

_METRIC_UNITS: dict[str, str] = {
    "ev": "model income units",
    "cv": "model income units",
    "ev_pct_consumption": "% baseline consumption",
    "ev_pct_gdp": "% baseline GDP",
    "price_effect": "model income units",
    "factor_income_effect": "model income units",
    "real_gdp_growth_pct": "% real GDP growth",
    "cpi": "index",
    "cpi_inflation_pct": "% price change",
    "trade_balance": "net exports",
    "terms_of_trade": "index",
    "gdp": "model value units",
    "tariffs": "model value units",
}

_METRIC_WELFARE_DEFINITIONS: dict[str, str] = {
    "ev": "hicksian_equivalent_variation",
    "cv": "hicksian_compensating_variation",
    "ev_pct_consumption": "hicksian_ev_pct_baseline_consumption",
    "ev_pct_gdp": "hicksian_ev_pct_baseline_gdp",
    "price_effect": "hicksian_price_effect",
    "factor_income_effect": "hicksian_factor_income_effect",
    "real_gdp_growth_pct": "geary_khamis_real_gdp_growth_pct",
    "cpi": "wage_deflated_cpi_level",
    "cpi_inflation_pct": "wage_deflated_cpi_inflation_pct",
    "trade_balance": "net_exports",
    "terms_of_trade": "terms_of_trade_index",
    "gdp": "nominal_gdp",
    "tariffs": "tariff_revenue",
}


def _check_rate(rate: Any, where: str) -> float:
    if isinstance(rate, bool) or not isinstance(rate, Real):
        raise TypeError(f"Tariff rate for {where} must be a real number, got {rate!r}")
    value = float(rate)
    if math.isnan(value) or math.isinf(value) or value < 0.0:
        raise ValueError(f"Tariff rate for {where} must be a finite non-negative number, got {rate!r}")
    return value


@dataclass(frozen=True)
class TariffScenario(metaclass=abc.ABCMeta):
    """One importing country's ad valorem import tariffs, by exporter and sector.

    Attributes
    ----------
    importer : str
        Code of the country that levies the tariffs.
    tariffs : Mapping[tuple[str, str], float] or np.ndarray
        Rates (0.10 = 10%). A mapping is keyed by ``(exporter, sector)`` codes;
        the sector ``"*"`` stands for every sector of that exporter, and an
        explicit sector entry overrides it. An array has shape ``(nc, ns)``
        (exporter country by sector, in the calibration's order) and must be
        zero on the importer's own row. Flows not listed carry no tariff, and
        domestic flows are never tariffed.
    name : str, default "scenario"
        Label used in results and batch tables.
    description : str, default ""
        Free text.
    metadata : dict, default empty
        Free metadata, copied to the result's metadata.

    Examples
    --------
    >>> scen = TariffScenario("C01", {("C02", "*"): 0.10}, name="ten_percent_on_C02")
    """

    importer: str
    tariffs: Mapping[tuple[str, str], float] | np.ndarray
    name: str = "scenario"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.importer, str) or not self.importer:
            raise TypeError("importer must be a non-empty country code (str)")
        if not isinstance(self.name, str):
            raise TypeError("name must be a str")
        if isinstance(self.tariffs, np.ndarray):
            arr = np.array(self.tariffs, dtype=float)
            if arr.ndim != 2:
                raise ValueError(f"A tariff array must have shape (nc, ns); got {arr.shape}")
            if not np.isfinite(arr).all() or np.any(arr < 0):
                raise ValueError("Tariff rates in the array must be finite and non-negative")
            arr.setflags(write=False)
            object.__setattr__(self, "tariffs", arr)
        elif isinstance(self.tariffs, Mapping):
            rates: dict[tuple[str, str], float] = {}
            for key, rate in self.tariffs.items():
                if (not isinstance(key, tuple) or len(key) != 2
                        or not all(isinstance(k, str) for k in key)):
                    raise TypeError(f"Tariff keys must be (exporter, sector) code pairs; got {key!r}")
                rates[key] = _check_rate(rate, f"{key}")
            object.__setattr__(self, "tariffs", rates)
        else:
            raise TypeError(
                "tariffs must be a mapping {(exporter, sector): rate} or an (nc, ns) array of rates"
            )

    def rate_matrix(self, country_codes: Sequence[str], sector_codes: Sequence[str]) -> np.ndarray:
        """Rates as an ``(nc, ns)`` array (exporter by sector) for the given code lists.

        Raises ``ValueError`` for an unknown importer, exporter or sector code, a
        tariff on the importer's own goods, or an array of the wrong shape.
        """
        _, rates, _ = _tariff_scenario_mask_and_rates(self, country_codes, sector_codes)
        return rates


def _tariff_scenario_mask_and_rates(
    scenario: TariffScenario,
    country_codes: Sequence[str],
    sector_codes: Sequence[str],
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return ``(importer_idx, rates_2d, written_2d)`` for one ``TariffScenario``.

    Mapping entries mark ``written_2d[o_idx, s_idx] = True`` even when the
    specified rate is ``0.0`` (preserving explicit FTA carve-outs when composed),
    whereas array inputs mark ``written_2d = rates_2d != 0.0``.
    """
    countries = [str(c) for c in country_codes]
    sectors = [str(s) for s in sector_codes]
    nc, ns = len(countries), len(sectors)
    if scenario.importer not in countries:
        raise ValueError(
            f"Unknown country code {scenario.importer!r} (importer); known codes: {countries}"
        )
    d_idx = countries.index(scenario.importer)
    if isinstance(scenario.tariffs, np.ndarray):
        if scenario.tariffs.shape != (nc, ns):
            raise ValueError(
                f"Tariff array has shape {scenario.tariffs.shape}; "
                f"this calibration needs (nc, ns) = ({nc}, {ns})"
            )
        rates = np.array(scenario.tariffs, dtype=float)
        written = rates != 0.0
    else:
        rates = np.zeros((nc, ns), dtype=float)
        written = np.zeros((nc, ns), dtype=bool)
        explicit: list[tuple[int, int, float]] = []
        for (exporter, sector), rate in scenario.tariffs.items():
            if exporter not in countries:
                raise ValueError(
                    f"Unknown country code {exporter!r} (exporter); known codes: {countries}"
                )
            o_idx = countries.index(exporter)
            if sector == ALL_SECTORS:
                rates[o_idx, :] = rate
                written[o_idx, :] = True
            elif sector in sectors:
                explicit.append((o_idx, sectors.index(sector), rate))
            else:
                raise ValueError(f"Unknown sector code {sector!r}; known codes: {sectors}")
        for o_idx, s_idx, rate in explicit:
            rates[o_idx, s_idx] = rate
            written[o_idx, s_idx] = True
    if np.any(rates[d_idx] != 0.0):
        raise ValueError(
            f"Domestic flows are never tariffed: {scenario.importer!r} lists a rate on its own goods"
        )
    written[d_idx, :] = False
    return d_idx, rates, written


def _layer_rates_and_written(
    scenario: TariffScenario | CompositeTariffScenario,
    country_codes: Sequence[str],
    sector_codes: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve one scenario layer to ``(rates_3d, written_3d)`` of shape ``(nc, nc, ns)``."""
    nc = len(country_codes)
    ns = len(sector_codes)
    if isinstance(scenario, CompositeTariffScenario):
        return scenario._resolve_rates_and_written(country_codes, sector_codes, allow_error=True)
    d_idx, rates_2d, written_2d = _tariff_scenario_mask_and_rates(
        scenario, country_codes, sector_codes
    )
    rates_3d = np.zeros((nc, nc, ns), dtype=float)
    written_3d = np.zeros((nc, nc, ns), dtype=bool)
    rates_3d[:, d_idx, :] = rates_2d
    written_3d[:, d_idx, :] = written_2d
    return rates_3d, written_3d


def _apply_conflict_mode(
    combined: np.ndarray,
    has_written: np.ndarray,
    layer_rates: np.ndarray,
    layer_written: np.ndarray,
    mode: ConflictMode,
    countries: Sequence[str],
    sectors: Sequence[str],
    layer_name: str,
) -> None:
    """Merge ``layer_rates`` (where ``layer_written`` is True) into ``combined``."""
    overlap = has_written & layer_written
    if mode == "error" and np.any(overlap):
        coords = np.argwhere(overlap)[:5]
        samples = [
            f"({countries[o]!r}, {countries[d]!r}, {sectors[s]!r})"
            for o, d, s in coords
        ]
        n_total = int(np.sum(overlap))
        raise ValueError(
            f"Tariff conflict in layer {layer_name!r} across {n_total} bilateral cell(s): "
            + ", ".join(samples)
        )
    fresh = layer_written & ~has_written
    combined[fresh] = layer_rates[fresh]
    if np.any(overlap):
        if mode == "last_wins":
            combined[overlap] = layer_rates[overlap]
        elif mode == "first_wins":
            pass
        elif mode == "max":
            combined[overlap] = np.maximum(combined[overlap], layer_rates[overlap])
        elif mode == "sum":
            combined[overlap] = combined[overlap] + layer_rates[overlap]
    has_written |= layer_written


def _format_unique_layer_names(
    scenarios: Sequence[TariffScenario | CompositeTariffScenario],
) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    seen_counts: dict[str, int] = {}
    for scen in scenarios:
        counts[scen.name] = counts.get(scen.name, 0) + 1
    for scen in scenarios:
        if counts[scen.name] == 1:
            out.append(scen.name)
        else:
            idx = seen_counts.get(scen.name, 0) + 1
            seen_counts[scen.name] = idx
            out.append(f"{scen.name}#{idx}")
    return out


@dataclass(frozen=True)
class CompositeTariffScenario:
    """Multi-importer composition of :class:`TariffScenario` or nested composites.

    Layers component tariff schedules into a bilateral rate tensor
    ``rates[exporter, importer, sector]`` under explicit conflict-resolution
    rules, preserving explicit zero-rate carve-outs in mapping tariffs.

    Attributes
    ----------
    scenarios : tuple[TariffScenario | CompositeTariffScenario, ...]
        Component scenarios evaluated in sequence.
    name : str, default "composite"
        Identifier propagated to equilibrium results and batch tables.
    on_conflict : {"last_wins", "first_wins", "error", "max", "sum"}, default "last_wins"
        Rule applied when two component scenarios explicitly write the same
        ``(exporter, importer, sector)`` cell.
    description : str, default ""
        Optional free text describing the policy package.
    metadata : dict[str, Any], default empty
        Free metadata copied to equilibrium results.
    """

    scenarios: tuple[TariffScenario | CompositeTariffScenario, ...]
    name: str = "composite"
    on_conflict: ConflictMode = "last_wins"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a str")
        if self.on_conflict not in VALID_CONFLICT_MODES:
            raise ValueError(
                f"on_conflict must be one of {VALID_CONFLICT_MODES}; got {self.on_conflict!r}"
            )
        if isinstance(self.scenarios, (str, TariffScenario, CompositeTariffScenario)):
            raise TypeError("scenarios must be a sequence of TariffScenario or CompositeTariffScenario")
        if not isinstance(self.scenarios, Sequence):
            raise TypeError("scenarios must be a sequence of TariffScenario or CompositeTariffScenario")
        items: list[TariffScenario | CompositeTariffScenario] = []
        for item in self.scenarios:
            if isinstance(item, str):
                raise TypeError(NAMED_SCENARIOS_REMOVED)
            if not isinstance(item, (TariffScenario, CompositeTariffScenario)):
                raise TypeError(
                    "CompositeTariffScenario entries must be TariffScenario or CompositeTariffScenario, "
                    f"got {type(item).__name__}"
                )
            items.append(item)
        if not items:
            raise ValueError("CompositeTariffScenario requires at least one component scenario")
        object.__setattr__(self, "scenarios", tuple(items))
        object.__setattr__(self, "metadata", dict(self.metadata) if self.metadata is not None else {})

    @property
    def importers(self) -> tuple[str, ...]:
        """Unique importer codes across component scenarios in first-seen order."""
        ordered: list[str] = []
        for item in self.scenarios:
            item_importers = item.importers if isinstance(item, CompositeTariffScenario) else (item.importer,)
            for imp in item_importers:
                if imp not in ordered:
                    ordered.append(imp)
        return tuple(ordered)

    def _resolve_rates_and_written(
        self,
        country_codes: Sequence[str],
        sector_codes: Sequence[str],
        *,
        allow_error: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        countries = [str(c) for c in country_codes]
        sectors = [str(s) for s in sector_codes]
        nc, ns = len(countries), len(sectors)
        combined = np.zeros((nc, nc, ns), dtype=float)
        has_written = np.zeros((nc, nc, ns), dtype=bool)
        mode: ConflictMode = self.on_conflict if allow_error else "last_wins"
        for scen in self.scenarios:
            layer_rates, layer_written = _layer_rates_and_written(scen, countries, sectors)
            _apply_conflict_mode(
                combined, has_written, layer_rates, layer_written, mode, countries, sectors, scen.name
            )
        return combined, has_written

    def to_rate_tensor(
        self,
        country_codes: Sequence[str],
        sector_codes: Sequence[str],
    ) -> np.ndarray:
        """Resolved ``(nc, nc, ns)`` bilateral rate tensor ``[exporter, importer, sector]``."""
        rates, _ = self._resolve_rates_and_written(country_codes, sector_codes, allow_error=True)
        return rates

    def to_frame(
        self,
        country_codes: Sequence[str],
        sector_codes: Sequence[str],
        *,
        nonzero_only: bool = True,
    ) -> pd.DataFrame:
        """Return resolved tariffs in long ``(exporter, importer, sector, rate)`` form."""
        countries = [str(c) for c in country_codes]
        sectors = [str(s) for s in sector_codes]
        rates, _ = self._resolve_rates_and_written(countries, sectors, allow_error=True)
        rows: list[dict[str, Any]] = []
        for o_idx, exp_code in enumerate(countries):
            for d_idx, imp_code in enumerate(countries):
                if o_idx == d_idx:
                    continue
                for s_idx, sec_code in enumerate(sectors):
                    rate = float(rates[o_idx, d_idx, s_idx])
                    if nonzero_only and rate == 0.0:
                        continue
                    rows.append({
                        "exporter": exp_code,
                        "importer": imp_code,
                        "sector": sec_code,
                        "rate": rate,
                    })
        return pd.DataFrame(rows, columns=["exporter", "importer", "sector", "rate"])

    def conflicts(
        self,
        country_codes: Sequence[str],
        sector_codes: Sequence[str],
    ) -> pd.DataFrame:
        """List bilateral ``(exporter, importer, sector)`` cells written by 2+ components.

        Inspects component layers even when ``on_conflict='error'`` (recording
        ``NaN`` for ``resolved_rate`` when ``on_conflict == 'error'``).
        """
        countries = [str(c) for c in country_codes]
        sectors = [str(s) for s in sector_codes]
        nc, ns = len(countries), len(sectors)
        layer_names = _format_unique_layer_names(self.scenarios)
        layers = [_layer_rates_and_written(s, countries, sectors) for s in self.scenarios]
        write_counts = np.sum([w.astype(int) for _, w in layers], axis=0)
        resolved: np.ndarray | None
        if self.on_conflict == "error":
            resolved = None
        else:
            resolved, _ = self._resolve_rates_and_written(countries, sectors, allow_error=True)

        rows: list[dict[str, Any]] = []
        coords = np.argwhere(write_counts >= 2)
        for o_idx, d_idx, s_idx in coords:
            sources: dict[str, float] = {}
            for lname, (lr, lw) in zip(layer_names, layers):
                if lw[o_idx, d_idx, s_idx]:
                    sources[lname] = float(lr[o_idx, d_idx, s_idx])
            rows.append({
                "exporter": countries[int(o_idx)],
                "importer": countries[int(d_idx)],
                "sector": sectors[int(s_idx)],
                "n_sources": len(sources),
                "sources": sources,
                "resolved_rate": (
                    np.nan if resolved is None else float(resolved[int(o_idx), int(d_idx), int(s_idx)])
                ),
                "rule": self.on_conflict,
            })
        cols = ["exporter", "importer", "sector", "n_sources", "sources", "resolved_rate", "rule"]
        return pd.DataFrame(rows, columns=cols)


# Register CompositeTariffScenario as a virtual TariffScenario so frozen
# solver entry points checking ``isinstance(x, TariffScenario)`` accept
# composites without altering frozen reference files.
TariffScenario.register(CompositeTariffScenario)


def combine_scenarios(
    scenarios: Sequence[TariffScenario | CompositeTariffScenario],
    *,
    name: str = "composite",
    on_conflict: ConflictMode = "last_wins",
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> CompositeTariffScenario:
    """Compose multiple :class:`TariffScenario` or composite schedules into one package.

    Parameters
    ----------
    scenarios : Sequence[TariffScenario | CompositeTariffScenario]
        Component schedules applied in sequence.
    name : str, default "composite"
        Identifier for the resulting package.
    on_conflict : {"last_wins", "first_wins", "error", "max", "sum"}, default "last_wins"
        Conflict-resolution policy when multiple layers specify rates on the same
        ``(exporter, importer, sector)`` cell.
    description : str, default ""
        Optional description.
    metadata : dict, optional
        Optional metadata dictionary attached to the scenario.
    """
    if isinstance(scenarios, (str, TariffScenario, CompositeTariffScenario)):
        raise TypeError("scenarios must be a sequence of TariffScenario or CompositeTariffScenario")
    if not isinstance(scenarios, Sequence):
        raise TypeError("scenarios must be a sequence of TariffScenario or CompositeTariffScenario")
    return CompositeTariffScenario(
        scenarios=tuple(scenarios),
        name=name,
        on_conflict=on_conflict,
        description=description,
        metadata={} if metadata is None else dict(metadata),
    )


def _require_scenario(scenario: Any) -> TariffScenario | CompositeTariffScenario:
    if isinstance(scenario, str):
        raise TypeError(NAMED_SCENARIOS_REMOVED)
    if not isinstance(scenario, (TariffScenario, CompositeTariffScenario)):
        raise TypeError(
            "scenario must be a TariffScenario or CompositeTariffScenario, "
            f"got {type(scenario).__name__}"
        )
    return scenario


def build_tariff_matrices(
    calib: TradeCalibrationResult,
    scenario: TariffScenario | CompositeTariffScenario,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilateral tariff multipliers ``1 + rate`` of one single or composite scenario.

    Parameters
    ----------
    calib : TradeCalibrationResult
        Calibration whose dimensions and codes index the tensors.
    scenario : TariffScenario or CompositeTariffScenario
        The scenario. A string raises ``TypeError`` (named scenarios were removed).

    Returns
    -------
    tau : np.ndarray, shape (ns, nc, ns, nc)
        Intermediate multipliers ``tau[s_orig, c_orig, s_dest, c_dest]``.
    tau_fd : np.ndarray, shape (ns, nc, nfd, nc)
        Final-demand multipliers ``tau_fd[s_orig, c_orig, fd_dest, c_dest]``.

    Both apply each importer's rate on goods of ``(c_orig, s_orig)`` for every
    using sector and final-use category; every other entry, including all
    domestic flows, is one. ``solve_trade_equilibrium`` accepts both arrays as
    given.
    """
    scenario = _require_scenario(scenario)
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")
    nc, ns, nfd = calib.n_countries, calib.n_sectors, calib.n_final_demand

    if isinstance(scenario, CompositeTariffScenario):
        rates_3d = scenario.to_rate_tensor(calib.country_codes, calib.sector_codes)
        mult_3d = 1.0 + np.transpose(rates_3d, (2, 0, 1))  # (ns, nc_orig, nc_dest)
        tau = np.broadcast_to(mult_3d[:, :, None, :], (ns, nc, ns, nc)).copy()
        tau_fd = np.broadcast_to(mult_3d[:, :, None, :], (ns, nc, nfd, nc)).copy()
        diag_idx = np.arange(nc)
        tau[:, diag_idx, :, diag_idx] = 1.0
        tau_fd[:, diag_idx, :, diag_idx] = 1.0
        return tau, tau_fd

    rates = scenario.rate_matrix(calib.country_codes, calib.sector_codes)
    d_idx = list(calib.country_codes).index(scenario.importer)
    tau = np.ones((ns, nc, ns, nc), dtype=float)
    tau_fd = np.ones((ns, nc, nfd, nc), dtype=float)
    mult = 1.0 + rates.T  # (ns, nc): origin sector by origin country
    tau[:, :, :, d_idx] = mult[:, :, None]
    tau_fd[:, :, :, d_idx] = mult[:, :, None]
    tau[:, d_idx, :, d_idx] = 1.0
    tau_fd[:, d_idx, :, d_idx] = 1.0
    return tau, tau_fd


def run_scenario(
    calib: TradeCalibrationResult,
    scenario: TariffScenario | CompositeTariffScenario,
    **solver_opts: Any,
) -> TradeEquilibriumResult:
    """Solve the equilibrium of one tariff scenario or composite package.

    ``solver_opts`` go to :func:`puremacro.trade.solver.solve_trade_equilibrium`
    (``method``, ``tol``, ``x0``, ``base_result``, ...). The result's metadata
    records ``scenario_name``, ``scenario_importers``, and ``scenario_importer``
    (set when there is a single importer, or ``None`` for multi-importer mixes).
    """
    scenario = _require_scenario(scenario)
    for key in ("tau", "tau_fd", "tauf", "tauf_fd"):
        if key in solver_opts:
            raise TypeError(f"run_scenario takes its tariffs from the scenario; do not pass {key}=")
    tau, tau_fd = build_tariff_matrices(calib, scenario)
    res = solve_trade_equilibrium(calib, tau=tau, tau_fd=tau_fd, **solver_opts)
    importers = (
        scenario.importers if isinstance(scenario, CompositeTariffScenario) else (scenario.importer,)
    )
    res.metadata["scenario_name"] = scenario.name
    res.metadata["scenario_importers"] = importers
    res.metadata["scenario_importer"] = importers[0] if len(importers) == 1 else None
    if scenario.metadata:
        res.metadata["scenario_metadata"] = dict(scenario.metadata)
    return res


def run_scenario_batch(
    calib: TradeCalibrationResult,
    scenarios: Sequence[TariffScenario | CompositeTariffScenario],
    *,
    baseline_name: str = "baseline",
    base_result: TradeEquilibriumResult | None = None,
    warm_start: bool = True,
    compute_gk: bool = True,
    numeraire: str | None = None,
    **solver_opts: Any,
) -> ScenarioBatchResult:
    """Solve a baseline and several tariff scenarios, warm-starting each solve.

    The baseline (no tariffs) is solved first under ``baseline_name`` unless
    ``base_result`` supplies it. Each scenario then starts from the previous
    solution when ``warm_start`` is true (from the calibrated state otherwise).
    With ``compute_gk`` the Geary-Khamis PPP and real GDP of every state are
    computed against the baseline (``numeraire`` is passed to
    :func:`~puremacro.trade.geary_khamis.compute_geary_khamis`).

    ``solver_opts`` go to every solve. Scenario names must be distinct and
    differ from ``baseline_name``.
    """
    if isinstance(scenarios, (str, TariffScenario, CompositeTariffScenario)):
        raise TypeError("scenarios must be a sequence of TariffScenario or CompositeTariffScenario objects")
    scen_list = [_require_scenario(s) for s in scenarios]
    names = [s.name for s in scen_list]
    if len(set(names)) != len(names) or baseline_name in names:
        raise ValueError(f"Scenario names must be distinct and differ from {baseline_name!r}; got {names}")
    if "x0" in solver_opts:
        raise TypeError("run_scenario_batch chooses the starting points; do not pass x0=")

    results: dict[str, TradeEquilibriumResult] = {}
    if base_result is None:
        base_result = solve_trade_equilibrium(calib, **solver_opts)
    results[baseline_name] = base_result
    base_converged = bool(base_result.converged)
    base_x_ok = bool(base_converged and np.isfinite(base_result.x_sol).all())
    if not base_converged:
        warnings.warn(
            f"Baseline scenario {baseline_name!r} did not converge "
            f"(max_residual={base_result.max_residual:.3e}); skipping warm-start reuse "
            "and Geary-Khamis PPP computation.",
            RuntimeWarning,
            stacklevel=2,
        )
    prev_x = base_result.x_sol.copy() if base_x_ok else build_initial_guess(calib)

    for scen in scen_list:
        x_init = prev_x.copy() if warm_start else build_initial_guess(calib)
        res = run_scenario(
            calib,
            scen,
            x0=x_init,
            base_result=base_result if base_converged else None,
            **solver_opts,
        )
        results[scen.name] = res
        if not res.converged:
            warnings.warn(
                f"Scenario {scen.name!r} did not converge "
                f"(max_residual={res.max_residual:.3e}); skipping warm-start update "
                "and Geary-Khamis PPP computation.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif warm_start and np.isfinite(res.x_sol).all():
            prev_x = res.x_sol.copy()

    gk_results: dict[str, GearyKhamisResult] = {}
    if compute_gk and base_converged:
        for name, res in results.items():
            if not res.converged:
                continue
            gk_res = compute_geary_khamis(
                res,
                base_result,
                calib,
                method="iterative",
                numeraire=numeraire,
                tol=1e-10,
                max_iter=1000,
            )
            if not gk_res.converged:
                warnings.warn(
                    f"Geary-Khamis PPP did not converge for scenario {name!r}; "
                    "omitting from Geary-Khamis real GDP growth results.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            gk_results[name] = gk_res

    batch = ScenarioBatchResult(
        scenarios=results,
        geary_khamis=gk_results,
        baseline_scenario=baseline_name,
        country_codes=tuple(calib.country_codes),
        sector_codes=tuple(calib.sector_codes),
        metadata={
            "warm_start": warm_start,
            "numeraire": numeraire,
            "solver_options": {k: v for k, v in solver_opts.items() if k != "base_result"},
            "converged": {name: bool(res.converged) for name, res in results.items()},
        },
    )
    unconverged_names = {name for name, res in results.items() if not res.converged}
    if not base_converged or unconverged_names or (compute_gk and len(gk_results) != len(results)):
        real_gdp_table = batch.real_gdp_table.copy()
        if not base_converged:
            real_gdp_table.loc[:, :] = np.nan
        else:
            for name in results:
                if name in unconverged_names or (compute_gk and name not in gk_results):
                    if name in real_gdp_table.columns:
                        real_gdp_table[name] = np.nan
        batch = replace(batch, real_gdp_table=real_gdp_table)
    return batch


def _resolve_welfare_tol(
    welfare_tol: float | Literal["auto"],
    res: TradeEquilibriumResult,
    base_res: TradeEquilibriumResult,
) -> float:
    if welfare_tol == "auto":
        r_tol = float(res.metadata.get("tol", 1e-8)) if isinstance(res.metadata, Mapping) else 1e-8
        b_tol = (
            float(base_res.metadata.get("tol", 1e-8)) if isinstance(base_res.metadata, Mapping) else 1e-8
        )
        return max(1e-8, r_tol, b_tol)
    if isinstance(welfare_tol, bool) or not isinstance(welfare_tol, Real) or float(welfare_tol) <= 0.0:
        raise ValueError(f"welfare_tol must be 'auto' or a positive number, got {welfare_tol!r}")
    return float(welfare_tol)


def _extract_array_scalar(arr: np.ndarray | None, idx: int) -> float:
    if arr is None:
        return np.nan
    flat = np.asarray(arr, dtype=float).ravel()
    if idx >= flat.size:
        return np.nan
    return float(flat[idx])


def _extract_batch_macro(
    batch: ScenarioBatchResult,
    res: TradeEquilibriumResult,
    scen_name: str,
    country: str,
    c_idx: int,
    base_name: str,
    has_nonzero_gdp_table: bool,
    *,
    base_converged: bool = True,
) -> dict[str, float]:
    res_converged = bool(res.converged)
    if not res_converged or not base_converged:
        real_gdp_growth_pct = np.nan
    elif scen_name in batch.geary_khamis:
        gk = batch.geary_khamis[scen_name]
        if not bool(gk.converged):
            real_gdp_growth_pct = np.nan
        else:
            growth = gk.gdp_growth if gk.gdp_growth is not None else gk.real_gdp_growth
            real_gdp_growth_pct = (
                _extract_array_scalar(growth, c_idx) if growth is not None else np.nan
            )
    elif (
        not batch.real_gdp_table.empty
        and country in batch.real_gdp_table.index
        and scen_name in batch.real_gdp_table.columns
        and (batch.geary_khamis or has_nonzero_gdp_table)
    ):
        real_gdp_growth_pct = float(batch.real_gdp_table.loc[country, scen_name])
    elif scen_name == base_name:
        real_gdp_growth_pct = 0.0
    else:
        real_gdp_growth_pct = np.nan

    if (
        not batch.cpi_table.empty
        and country in batch.cpi_table.index
        and scen_name in batch.cpi_table.columns
    ):
        cpi = float(batch.cpi_table.loc[country, scen_name])
    else:
        cpi = _extract_array_scalar(res.cpi, c_idx)
        if not np.isfinite(cpi) and scen_name == base_name:
            cpi = 1.0
    cpi_inflation_pct = (cpi - 1.0) * 100.0 if np.isfinite(cpi) else np.nan

    if (
        not batch.trade_balance_table.empty
        and country in batch.trade_balance_table.index
        and scen_name in batch.trade_balance_table.columns
    ):
        trade_balance = float(batch.trade_balance_table.loc[country, scen_name])
    else:
        exp = _extract_array_scalar(res.exports, c_idx)
        imp = _extract_array_scalar(res.imports, c_idx)
        trade_balance = exp - imp if (np.isfinite(exp) and np.isfinite(imp)) else np.nan

    return {
        "real_gdp_growth_pct": real_gdp_growth_pct,
        "cpi": cpi,
        "cpi_inflation_pct": cpi_inflation_pct,
        "trade_balance": trade_balance,
        "terms_of_trade": _extract_array_scalar(res.terms_of_trade, c_idx),
        "gdp": _extract_array_scalar(res.gdp, c_idx),
        "tariffs": _extract_array_scalar(res.tariffs, c_idx),
    }


def _build_scenario_summary_df(
    table: pd.DataFrame,
    scenario_order: Sequence[str],
    base_name: str,
    *,
    _warn_on_failure: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scen_name in scenario_order:
        sub = table[table["scenario"] == scen_name]
        if sub.empty:
            rows.append({
                "scenario": scen_name,
                "is_baseline": scen_name == base_name,
                "converged": False,
                "n_countries": 0,
                "n_welfare_passed": 0,
                "mean_ev_pct_consumption": np.nan,
                "min_ev_pct_consumption": np.nan,
                "total_ev": np.nan,
                "mean_real_gdp_growth_pct": np.nan,
                "mean_cpi_inflation_pct": np.nan,
            })
            continue

        conv_mask = sub["converged"].astype(bool).to_numpy()
        all_converged = bool(np.all(conv_mask))
        welf_mask = (sub["welfare_status"] == "passed").to_numpy() & conv_mask
        n_welfare_passed = int(np.sum(welf_mask))
        all_welfare_passed = bool(all_converged and np.all(welf_mask))

        if _warn_on_failure:
            if not all_converged:
                warnings.warn(
                    f"Scenario {scen_name!r} did not converge; setting summary "
                    "welfare and macro aggregates to NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif not all_welfare_passed:
                warnings.warn(
                    f"Scenario {scen_name!r} has failed welfare status for one or more "
                    "countries; setting welfare summary aggregates to NaN.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        ev_pct = sub["ev_pct_consumption"].to_numpy(dtype=float)
        ev_abs = sub["ev"].to_numpy(dtype=float)
        gdp_g = sub["real_gdp_growth_pct"].to_numpy(dtype=float)
        infl = sub["cpi_inflation_pct"].to_numpy(dtype=float)

        welfare_ok = (
            all_welfare_passed
            and bool(np.all(np.isfinite(ev_pct)))
            and bool(np.all(np.isfinite(ev_abs)))
        )
        gdp_ok = all_converged and bool(np.all(np.isfinite(gdp_g)))
        infl_ok = all_converged and bool(np.all(np.isfinite(infl)))

        rows.append({
            "scenario": scen_name,
            "is_baseline": scen_name == base_name,
            "converged": all_converged,
            "n_countries": len(sub),
            "n_welfare_passed": n_welfare_passed,
            "mean_ev_pct_consumption": float(np.mean(ev_pct)) if welfare_ok else np.nan,
            "min_ev_pct_consumption": float(np.min(ev_pct)) if welfare_ok else np.nan,
            "total_ev": float(np.sum(ev_abs)) if welfare_ok else np.nan,
            "mean_real_gdp_growth_pct": float(np.mean(gdp_g)) if gdp_ok else np.nan,
            "mean_cpi_inflation_pct": float(np.mean(infl)) if infl_ok else np.nan,
        })
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class ScenarioComparisonResult:
    """Structured Hicksian welfare and macro comparison across trade scenarios.

    Attributes
    ----------
    table : pd.DataFrame
        Country-by-scenario long table containing welfare and macro indicators.
    scenario_summary : pd.DataFrame
        Scenario-level aggregate table across evaluated countries.
    baseline : str
        Name of the baseline benchmark scenario.
    countries : tuple[str, ...]
        Country codes included in the comparison.
    scenarios : tuple[str, ...]
        Scenario names included in the comparison.
    numeraire : str | None
        Numeraire code recorded from the batch (if any).
    warnings : tuple[str, ...]
        Non-fatal notices collected during validation or aggregation.
    digits : int
        Default rounding precision for table export methods.
    """

    table: pd.DataFrame
    scenario_summary: pd.DataFrame
    baseline: str
    countries: tuple[str, ...]
    scenarios: tuple[str, ...]
    numeraire: str | None = None
    warnings: tuple[str, ...] = ()
    digits: int = 3

    def to_frame(
        self,
        layout: Literal["full", "compact", "summary"] = "full",
    ) -> pd.DataFrame:
        """Return comparison table in ``'full'``, ``'compact'``, or ``'summary'`` layout."""
        if layout == "full":
            return self.table.copy()
        if layout == "compact":
            cols = [
                "scenario",
                "country",
                "converged",
                "welfare_status",
                "ev_pct_consumption",
                "ev_pct_gdp",
                "real_gdp_growth_pct",
                "cpi_inflation_pct",
                "trade_balance",
            ]
            keep = [c for c in cols if c in self.table.columns]
            return self.table.loc[:, keep].copy()
        if layout == "summary":
            return self.scenario_summary.copy()
        raise ValueError(f"Unknown layout {layout!r}; expected 'full', 'compact', or 'summary'")

    def to_dataframe(
        self,
        layout: Literal["full", "compact", "summary"] = "full",
    ) -> pd.DataFrame:
        """Alias for :meth:`to_frame`."""
        return self.to_frame(layout=layout)

    def _filtered_table(self, include_baseline: bool) -> pd.DataFrame:
        if include_baseline:
            return self.table
        return self.table[self.table["scenario"] != self.baseline]

    def _ordered_pivot(
        self,
        df: pd.DataFrame,
        value_col: str,
        include_baseline: bool,
    ) -> pd.DataFrame:
        row_order = [s for s in self.scenarios if include_baseline or s != self.baseline]
        col_order = [c for c in self.countries if c in set(df["country"])]
        if df.empty or not row_order or not col_order:
            return pd.DataFrame(index=pd.Index(row_order, name="scenario"), columns=col_order)
        piv = df.pivot(index="scenario", columns="country", values=value_col)
        piv = piv.reindex(index=row_order, columns=col_order)
        piv.index.name = "scenario"
        piv.columns.name = "country"
        return piv

    def to_payoff_matrix(
        self,
        metric: str = "ev_pct_consumption",
        *,
        include_baseline: bool = True,
    ) -> pd.DataFrame:
        """Pivot one numeric indicator to a ``scenario x country`` payoff matrix."""
        non_metric = {"scenario", "country", "is_baseline", "converged", "welfare_status"}
        if metric in non_metric or metric not in self.table.columns:
            valid = [c for c in self.table.columns if c not in non_metric]
            raise ValueError(f"Unknown payoff metric {metric!r}; available metrics: {valid}")
        sub = self._filtered_table(include_baseline)
        piv = self._ordered_pivot(sub, metric, include_baseline)
        return piv.astype(float)

    def _status_matrix(
        self,
        metric: str,
        payoffs: pd.DataFrame,
        *,
        include_baseline: bool = True,
    ) -> pd.DataFrame:
        sub = self._filtered_table(include_baseline)
        conv_df = self._ordered_pivot(sub, "converged", include_baseline)
        welf_df = self._ordered_pivot(sub, "welfare_status", include_baseline)
        out = pd.DataFrame("failed", index=payoffs.index, columns=payoffs.columns, dtype=object)
        require_welfare = metric in _WELFARE_METRICS
        for scen in payoffs.index:
            for ctry in payoffs.columns:
                val = payoffs.loc[scen, ctry]
                conv_ok = bool(conv_df.loc[scen, ctry]) if pd.notna(conv_df.loc[scen, ctry]) else False
                welf_ok = (welf_df.loc[scen, ctry] == "passed") if require_welfare else True
                if conv_ok and welf_ok and np.isfinite(float(val)):
                    out.loc[scen, ctry] = "passed"
        return out

    def compare_robust(
        self,
        metric: str = "ev_pct_consumption",
        *,
        probabilities: Mapping[str, float] | pd.Series | None = None,
        include_baseline: bool = True,
        maximize: bool = True,
        comparability: Mapping[str, str] | None = None,
        tolerance: float = 1e-10,
    ) -> RobustPolicyComparison:
        """Evaluate scenario robustness across countries via :func:`~puremacro.study.robust.compare_policy_payoffs`."""
        from puremacro.study.robust import compare_policy_payoffs

        payoffs = self.to_payoff_matrix(metric=metric, include_baseline=include_baseline)
        if payoffs.empty:
            raise ValueError("No scenarios available for robust policy comparison")
        statuses = self._status_matrix(metric, payoffs, include_baseline=include_baseline)
        numeraire_label = (
            self.numeraire
            if isinstance(self.numeraire, str) and self.numeraire.strip()
            else "model_numeraire"
        )
        default_comp = {
            "population": f"representative_households({','.join(str(c) for c in payoffs.columns)})",
            "horizon": "comparative_static",
            "numeraire": numeraire_label,
            "welfare_definition": _METRIC_WELFARE_DEFINITIONS.get(metric, metric),
        }
        comp_map = (
            default_comp
            if comparability is None
            else {str(k): str(v) for k, v in comparability.items()}
        )
        prob_series = (
            probabilities
            if probabilities is None or isinstance(probabilities, pd.Series)
            else pd.Series({str(k): float(v) for k, v in probabilities.items()}, dtype=float)
        )
        units = _METRIC_UNITS.get(metric, metric)
        return compare_policy_payoffs(
            payoffs,
            objective=metric,
            units=units,
            comparability=comp_map,
            probabilities=prob_series,
            maximize=maximize,
            statuses=statuses,
            tolerance=tolerance,
        )

    def summary(self) -> str:
        """Return a formatted text overview of the scenario comparison."""
        header = (
            f"Scenario Comparison (baseline={self.baseline!r}, "
            f"{len(self.scenarios)} scenarios x {len(self.countries)} countries)"
        )
        lines = [
            header,
            self.scenario_summary.to_string(index=False),
            "",
            self.to_frame("compact").to_string(index=False),
        ]
        if self.warnings:
            lines.append("Warnings:")
            lines.extend(f"  - {w}" for w in self.warnings)
        return "\n".join(lines)

    def to_markdown(
        self,
        layout: Literal["full", "compact", "summary"] = "compact",
        digits: int | None = None,
    ) -> str:
        """Render the selected table layout as GitHub-Flavored Markdown."""
        d = self.digits if digits is None else int(digits)
        return df_to_markdown(self.to_frame(layout=layout), index=False, digits=d)

    def to_latex(
        self,
        layout: Literal["full", "compact", "summary"] = "compact",
        digits: int | None = None,
    ) -> str:
        """Render the selected table layout as LaTeX tabular markup."""
        d = self.digits if digits is None else int(digits)
        return df_to_latex(self.to_frame(layout=layout), index=False, digits=d)

    def to_typst(
        self,
        layout: Literal["full", "compact", "summary"] = "compact",
        digits: int | None = None,
    ) -> str:
        """Render the selected table layout as a Typst ``#table``."""
        d = self.digits if digits is None else int(digits)
        return df_to_typst(self.to_frame(layout=layout), index=False, digits=d)

    def plot(
        self,
        metric: str = "ev_pct_consumption",
        *,
        include_baseline: bool = True,
        ax: Axes | None = None,
        title: str | None = None,
    ) -> Figure | Axes:
        """Plot a grouped country bar chart across scenarios for ``metric``."""
        import matplotlib.pyplot as plt

        payoffs = self.to_payoff_matrix(metric=metric, include_baseline=include_baseline)
        if payoffs.empty:
            raise ValueError("No scenarios available to plot")

        fig: Figure | None = None
        if ax is None:
            fig, plot_ax = plt.subplots(figsize=(max(6.0, 1.4 * len(payoffs.columns)), 4.2))
        else:
            plot_ax = ax

        x = np.arange(len(payoffs.columns), dtype=float)
        n_scen = max(len(payoffs.index), 1)
        width = 0.8 / n_scen
        offsets = (np.arange(n_scen, dtype=float) - 0.5 * (n_scen - 1)) * width
        for i, scen_name in enumerate(payoffs.index):
            raw_vals = payoffs.loc[scen_name].to_numpy(dtype=float)
            plot_vals = np.where(np.isfinite(raw_vals), raw_vals, 0.0)
            label = str(scen_name) if str(scen_name) else "_nolegend_"
            plot_ax.bar(x + offsets[i], plot_vals, width=width, label=label)

        plot_ax.axhline(0.0, linewidth=0.8, alpha=0.6)
        plot_ax.set_xticks(x)
        plot_ax.set_xticklabels([str(c) for c in payoffs.columns])
        plot_ax.set_xlabel("Country")
        plot_ax.set_ylabel(_METRIC_UNITS.get(metric, metric))
        plot_ax.set_title(title if title is not None else f"Scenario Comparison: {metric}")
        if n_scen > 1:
            plot_ax.legend()

        if fig is not None:
            fig.tight_layout()
            return fig
        return plot_ax


def summarize_scenario_batch(
    batch: ScenarioBatchResult,
    calib: TradeCalibrationResult,
    *,
    countries: Sequence[str] | None = None,
    scenarios: Sequence[str] | None = None,
    include_baseline: bool = True,
    consumption_categories: Sequence[int] = (0,),
    welfare_tol: float | Literal["auto"] = "auto",
    digits: int = 3,
) -> ScenarioComparisonResult:
    """Compute Hicksian welfare and macro comparison tables for a scenario batch.

    Parameters
    ----------
    batch : ScenarioBatchResult
        Output of :func:`run_scenario_batch`.
    calib : TradeCalibrationResult
        Baseline calibration matching ``batch``.
    countries : Sequence[str], optional
        Subset/order of country codes to include; defaults to all calibration countries.
    scenarios : Sequence[str], optional
        Subset/order of scenario names to include; defaults to all batch scenarios.
    include_baseline : bool, default True
        Whether to retain the baseline scenario row group (with exact zero welfare changes).
    consumption_categories : Sequence[int], default (0,)
        Final-demand indices passed to :func:`~puremacro.trade.welfare.compute_hicksian_welfare`.
    welfare_tol : float or "auto", default "auto"
        Residual check tolerance for welfare validation; ``"auto"`` uses
        ``max(1e-8, tol, base_tol)`` from solve metadata.
    digits : int, default 3
        Default formatting precision stored on the returned result.
    """
    from puremacro.trade.welfare import _checked_state, compute_hicksian_welfare

    if not isinstance(batch, ScenarioBatchResult):
        raise TypeError("batch must be a ScenarioBatchResult")
    if not isinstance(calib, TradeCalibrationResult):
        raise TypeError("calib must be a TradeCalibrationResult")

    cat_tuple = tuple(int(c) for c in consumption_categories)
    if not cat_tuple:
        raise ValueError("consumption_categories must contain at least one index")

    base_name = batch.baseline_scenario
    if base_name not in batch.scenarios:
        raise ValueError(f"Baseline scenario {base_name!r} not found in batch.scenarios")
    base_res = batch.scenarios[base_name]

    known_countries = (
        tuple(str(c) for c in calib.country_codes)
        if calib.country_codes
        else tuple(str(c) for c in batch.country_codes)
    )
    if not known_countries:
        known_countries = tuple(f"C{i:02d}" for i in range(calib.n_countries))

    if countries is None:
        target_countries = known_countries
    else:
        target_countries = tuple(str(c) for c in countries)
        for c in target_countries:
            if c not in known_countries:
                raise ValueError(f"Unknown country code {c!r}; known codes: {list(known_countries)}")
    if not target_countries:
        raise ValueError("countries must contain at least one country code")

    if scenarios is None:
        raw_scenarios = list(batch.scenarios.keys())
    else:
        raw_scenarios = [str(s) for s in scenarios]
        for s in raw_scenarios:
            if s not in batch.scenarios:
                raise ValueError(
                    f"Unknown scenario {s!r}; available scenarios: {list(batch.scenarios)}"
                )
    target_scenarios = tuple(
        s for s in raw_scenarios if include_baseline or s != base_name
    )
    if not target_scenarios:
        raise ValueError("No scenarios selected for summary")

    warn_list: list[str] = []
    base_converged = bool(base_res.converged)
    base_valid = base_converged
    base_tol = _resolve_welfare_tol(welfare_tol, base_res, base_res)
    if not base_valid:
        msg = (
            f"Baseline scenario {base_name!r} did not converge; "
            "welfare and real GDP growth metrics unavailable."
        )
        warn_list.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    else:
        try:
            _checked_state(base_res, calib, tol=base_tol)
        except (ValueError, RuntimeError, ArithmeticError, KeyError) as exc:
            base_valid = False
            msg = f"Baseline scenario {base_name!r} failed state check: {exc}"
            warn_list.append(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

    gdp_table_vals = np.asarray(batch.real_gdp_table, dtype=float)
    has_nonzero_gdp_table = (
        not batch.real_gdp_table.empty
        and bool(np.any(np.isfinite(gdp_table_vals) & (gdp_table_vals != 0.0)))
    )

    rows: list[dict[str, Any]] = []
    for scen_name in target_scenarios:
        res = batch.scenarios[scen_name]
        is_base = scen_name == base_name
        converged = bool(res.converged)
        scen_state_ok = converged and base_valid
        if not is_base:
            if not converged:
                msg = (
                    f"Scenario {scen_name!r} did not converge; "
                    "welfare and aggregate summary metrics set to NaN."
                )
                warn_list.append(msg)
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            elif base_valid:
                scen_tol = _resolve_welfare_tol(welfare_tol, res, base_res)
                try:
                    _checked_state(res, calib, tol=scen_tol)
                except (ValueError, RuntimeError, ArithmeticError, KeyError) as exc:
                    scen_state_ok = False
                    msg = f"Scenario {scen_name!r} failed welfare state check: {exc}"
                    warn_list.append(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)

        gk_res = batch.geary_khamis.get(scen_name)
        if converged and base_converged and gk_res is not None and not bool(gk_res.converged):
            msg = (
                f"Geary-Khamis PPP did not converge for scenario {scen_name!r}; "
                "real GDP growth set to NaN."
            )
            warn_list.append(msg)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        for ctry in target_countries:
            c_idx = known_countries.index(ctry)
            macro = _extract_batch_macro(
                batch,
                res,
                scen_name,
                ctry,
                c_idx,
                base_name,
                has_nonzero_gdp_table,
                base_converged=base_converged,
            )
            welfare_status = "failed"
            ev = cv = ev_pct_c = ev_pct_g = p_eff = f_eff = np.nan

            if is_base:
                if base_valid:
                    welfare_status = "passed"
                    ev = cv = ev_pct_c = ev_pct_g = p_eff = f_eff = 0.0
            elif scen_state_ok:
                tol_s = _resolve_welfare_tol(welfare_tol, res, base_res)
                try:
                    w = compute_hicksian_welfare(
                        calib,
                        res,
                        base_result=base_res,
                        target_country=c_idx,
                        consumption_categories=cat_tuple,
                        tol=tol_s,
                    )
                    vals = (
                        float(w.ev),
                        float(w.cv),
                        float(w.ev_pct_consumption),
                        float(w.ev_pct_gdp),
                        float(w.price_effect),
                        float(w.factor_income_effect),
                    )
                    if all(np.isfinite(vals)):
                        ev, cv, ev_pct_c, ev_pct_g, p_eff, f_eff = vals
                        welfare_status = "passed"
                    else:
                        msg = (
                            f"Non-finite Hicksian welfare values for ({scen_name!r}, {ctry!r}); "
                            "setting welfare metrics to NaN."
                        )
                        warn_list.append(msg)
                        warnings.warn(msg, RuntimeWarning, stacklevel=2)
                except (ValueError, RuntimeError, ArithmeticError, KeyError) as exc:
                    msg = f"Welfare computation failed for ({scen_name!r}, {ctry!r}): {exc}"
                    warn_list.append(msg)
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)

            rows.append({
                "scenario": scen_name,
                "country": ctry,
                "is_baseline": is_base,
                "converged": converged,
                "welfare_status": welfare_status,
                "ev": ev,
                "cv": cv,
                "ev_pct_consumption": ev_pct_c,
                "ev_pct_gdp": ev_pct_g,
                "price_effect": p_eff,
                "factor_income_effect": f_eff,
                **macro,
            })

    table = pd.DataFrame(rows)
    scenario_summary = _build_scenario_summary_df(
        table, target_scenarios, base_name, _warn_on_failure=False
    )
    numeraire = (
        batch.metadata.get("numeraire") if isinstance(batch.metadata, Mapping) else None
    )
    return ScenarioComparisonResult(
        table=table,
        scenario_summary=scenario_summary,
        baseline=base_name,
        countries=target_countries,
        scenarios=target_scenarios,
        numeraire=numeraire,
        warnings=tuple(warn_list),
        digits=digits,
    )


__all__ = [
    "CompositeTariffScenario",
    "ScenarioComparisonResult",
    "TariffScenario",
    "build_tariff_matrices",
    "combine_scenarios",
    "run_scenario",
    "run_scenario_batch",
    "summarize_scenario_batch",
]
