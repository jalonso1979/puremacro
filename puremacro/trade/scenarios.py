"""Declarative Tariff Scenario Engine and Sequential Batch Runner for puremacro.trade.

This module provides declarative scenario specifications, 4D bilateral tariff tensor
construction, and warm-started sequential batch execution reproducing the canonical
counterfactual simulations from Main77c_11s.m and published paper impact tables.

Conforms strictly to the puremacro Pyodide runtime contract: pure NumPy/SciPy/Pandas,
zero dev-dependencies in the runtime path, fully vectorized.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from puremacro.trade._results import (
    GearyKhamisResult,
    ScenarioBatchResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
)
from puremacro.trade.data import (
    CANONICAL_COUNTRY_CODES,
    CANONICAL_SECTOR_CODES,
    EU_COUNTRY_CODES,
)
from puremacro.trade.geary_khamis import compute_geary_khamis, restore_capital_formation
from puremacro.trade.solver import build_initial_guess, solve_trade_equilibrium


# ---------------------------------------------------------------------------
# Declarative Scenario Specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TariffScenario:
    """Declarative specification of a bilateral tariff counterfactual scenario.

    Attributes
    ----------
    name : str
        Unique scenario identifier (e.g. 'base', 't10', 't10_25').
    description : str, default ''
        Economic and policy description of the scenario.
    us_import_tariffs : dict[str, float], default empty
        Ad-valorem tariff rates applied by the US on imports from specific origin countries.
        Keyed by ISO-3 exporter country code (e.g. {'CAN': 0.25, 'CHN': 0.54}).
    default_us_tariff : float, default 0.0
        Universal fallback ad-valorem tariff rate applied by the US on non-exempt foreign partners.
    bilateral_tariffs : dict[tuple[str, str], float], default empty
        Bilateral tariff overrides keyed by (exporter_code, importer_code).
    sectoral_tariffs : dict[tuple[str, str, str], float], default empty
        Sector-specific tariff overrides keyed by (exporter_code, importer_code, sector_code).
    metadata : dict[str, Any], default empty
        Additional scenario metadata.
    """

    name: str
    description: str = ""
    us_import_tariffs: dict[str, float] = field(default_factory=dict)
    default_us_tariff: float = 0.0
    bilateral_tariffs: dict[tuple[str, str], float] = field(default_factory=dict)
    sectoral_tariffs: dict[tuple[str, str, str], float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate tariffs and normalize attributes."""
        # Support tariffs alias passed via metadata or kwargs
        if "tariffs" in self.metadata and not self.us_import_tariffs:
            t_dict = self.metadata["tariffs"]
            if isinstance(t_dict, dict):
                object.__setattr__(self, "us_import_tariffs", dict(t_dict))

        # Validate rates
        if self.default_us_tariff < 0.0 or math.isnan(self.default_us_tariff) or math.isinf(self.default_us_tariff):
            raise ValueError(f"Tariff rate for default_us_tariff must be a finite non-negative number, got {self.default_us_tariff}")

        for k, v in self.us_import_tariffs.items():
            if not isinstance(v, (int, float)) or v < 0.0 or math.isnan(v) or math.isinf(v):
                raise ValueError(f"Tariff rate for {k} must be a finite non-negative number, got {v}")

        for pair, v_pair in self.bilateral_tariffs.items():
            if not isinstance(v_pair, (int, float)) or v_pair < 0.0 or math.isnan(v_pair) or math.isinf(v_pair):
                raise ValueError(f"Tariff rate for bilateral pair {pair} must be a finite non-negative number, got {v_pair}")

        for sec_key, v_sec in self.sectoral_tariffs.items():
            if not isinstance(v_sec, (int, float)) or v_sec < 0.0 or math.isnan(v_sec) or math.isinf(v_sec):
                raise ValueError(f"Tariff rate for sectoral tuple {sec_key} must be a finite non-negative number, got {v_sec}")

    def get_rate(
        self,
        origin_code: str,
        dest_code: str = "USA",
        sector_code: str | None = None,
    ) -> float:
        """Return the ad-valorem tariff rate for an origin-destination-sector transaction."""
        # Domestic sales are never tariffed
        if origin_code == dest_code:
            return 0.0

        # 1. Check sector-specific overrides
        if sector_code is not None and (origin_code, dest_code, sector_code) in self.sectoral_tariffs:
            return float(self.sectoral_tariffs[(origin_code, dest_code, sector_code)])

        # 2. Check general bilateral overrides
        if (origin_code, dest_code) in self.bilateral_tariffs:
            return float(self.bilateral_tariffs[(origin_code, dest_code)])

        # 3. Check US import tariffs
        if dest_code == "USA":
            if origin_code in self.us_import_tariffs:
                return float(self.us_import_tariffs[origin_code])
            return float(self.default_us_tariff)

        return 0.0

    def _resolve_codes(
        self,
        ns: int,
        nc: int,
        country_codes: Sequence[str] | None,
        sector_codes: Sequence[str] | None,
    ) -> tuple[list[str], list[str]]:
        """Resolve the country and sector code lists for the solver-layout tensors.

        The canonical registries are used only when the dimensions match them
        exactly; any other calibration must pass its own codes, because mapping
        e.g. a 2-country model onto the first two canonical codes ('ARG', 'AUS')
        would silently produce a tariff-free tensor.
        """
        if country_codes is None:
            if nc != len(CANONICAL_COUNTRY_CODES):
                raise ValueError(
                    f"nc={nc} does not match the {len(CANONICAL_COUNTRY_CODES)} canonical country codes; "
                    "pass country_codes=calib.country_codes (or use build_tariff_matrices(scenario, calib))."
                )
            countries = list(CANONICAL_COUNTRY_CODES)
        else:
            countries = [str(c) for c in country_codes]
            if len(countries) != nc:
                raise ValueError(f"country_codes has {len(countries)} entries but nc={nc}.")

        if sector_codes is None:
            if ns == len(CANONICAL_SECTOR_CODES):
                sectors = list(CANONICAL_SECTOR_CODES)
            elif self.sectoral_tariffs:
                raise ValueError(
                    f"ns={ns} does not match the {len(CANONICAL_SECTOR_CODES)} canonical sector codes and the "
                    "scenario has sectoral_tariffs; pass sector_codes=calib.sector_codes."
                )
            else:
                sectors = [f"S{i:02d}" for i in range(ns)]  # unused: no sectoral overrides
        else:
            sectors = [str(c) for c in sector_codes]
            if len(sectors) != ns:
                raise ValueError(f"sector_codes has {len(sectors)} entries but ns={ns}.")
        return countries, sectors

    def _build_solver_tensor(
        self,
        ns: int,
        nc: int,
        n_dest: int,
        country_codes: Sequence[str] | None,
        sector_codes: Sequence[str] | None,
    ) -> np.ndarray:
        """Fill a (ns*nc, n_dest, nc) multiplier tensor in the solver's origin-major layout."""
        countries, sectors = self._resolve_codes(ns, nc, country_codes, sector_codes)
        out = np.ones((ns * nc, n_dest, nc), dtype=float)
        for d_idx, d_code in enumerate(countries):
            for o_idx, o_code in enumerate(countries):
                if o_idx == d_idx:
                    continue
                rate = self.get_rate(origin_code=o_code, dest_code=d_code)
                if rate != 0.0:
                    out[o_idx * ns : (o_idx + 1) * ns, :, d_idx] = 1.0 + rate
                for s_idx, s_code in enumerate(sectors):
                    sec_rate = self.get_rate(origin_code=o_code, dest_code=d_code, sector_code=s_code)
                    if sec_rate != rate:
                        out[o_idx * ns + s_idx, :, d_idx] = 1.0 + sec_rate
        return out

    def build_intermediate_tariffs(
        self,
        ns: int = 11,
        nc: int = 77,
        *,
        country_codes: Sequence[str] | None = None,
        sector_codes: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Construct the intermediate tariff multiplier tensor of shape (ns*nc, ns, nc).

        This is the solver layout ``tau_a[o_sector + ns*o_country, d_sector, d_country]``
        equal to ``build_tariff_matrices(scenario, calib)[0].transpose(1, 0, 2, 3)
        .reshape(ns*nc, ns, nc)``.

        Parameters
        ----------
        ns, nc : int
            Number of sectors and countries.
        country_codes, sector_codes : sequence of str, optional
            Codes indexing the calibration. Required unless ``nc`` (and, when the
            scenario has sectoral overrides, ``ns``) match the canonical registries;
            a ValueError is raised otherwise instead of returning an all-ones tensor.
        """
        return self._build_solver_tensor(ns, nc, ns, country_codes, sector_codes)

    def build_final_demand_tariffs(
        self,
        ns: int = 11,
        nc: int = 77,
        nfd: int = 3,
        *,
        country_codes: Sequence[str] | None = None,
        sector_codes: Sequence[str] | None = None,
    ) -> np.ndarray:
        """Construct the final demand tariff multiplier tensor of shape (ns*nc, nfd, nc).

        Same layout and code-resolution rules as :meth:`build_intermediate_tariffs`.
        """
        return self._build_solver_tensor(ns, nc, nfd, country_codes, sector_codes)


@dataclass(frozen=True)
class ExtendedTariffScenario(TariffScenario):
    """Declarative specification of an extended tariff counterfactual scenario.

    Extends :class:`TariffScenario` with configuration specifications for the
    four advanced CGE extensions:
    - Retaliation games (Extension A)
    - Dynamic J-curve transition (Extension B)
    - Fiscal revenue recycling (Extension C)
    - Upstream capacity bottlenecks (Extension D)
    """

    retaliation_config: Any | None = None
    dynamic_config: Any | None = None
    recycling_config: Any | None = None
    bottleneck_config: Any | None = None


# ---------------------------------------------------------------------------
# Pre-Defined Canonical Paper Scenarios
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, TariffScenario] = {
    "base": TariffScenario(
        name="base",
        description="Benchmark pre-shock calibration with 0% unilateral tariffs.",
        default_us_tariff=0.0,
        us_import_tariffs={},
    ),
    "t10": TariffScenario(
        name="t10",
        description="Uniform 10% US import tariff across all foreign trading partners.",
        default_us_tariff=0.10,
        us_import_tariffs={},
    ),
    "t10_25": TariffScenario(
        name="t10_25",
        description="Graduated tariff: 10% ROW, 25% on Canada, Mexico, China, and Hong Kong.",
        default_us_tariff=0.10,
        us_import_tariffs={"CAN": 0.25, "MEX": 0.25, "CHN": 0.25, "HKG": 0.25},
    ),
    "t10_54": TariffScenario(
        name="t10_54",
        description="10% ROW, 25% on Canada and Mexico, 54% on China and Hong Kong.",
        default_us_tariff=0.10,
        us_import_tariffs={"CAN": 0.25, "MEX": 0.25, "CHN": 0.54, "HKG": 0.54},
    ),
    "t10_75": TariffScenario(
        name="t10_75",
        description="Intermediate escalation: 10% ROW, 25% on Canada and Mexico, 75% on China and Hong Kong.",
        default_us_tariff=0.10,
        us_import_tariffs={"CAN": 0.25, "MEX": 0.25, "CHN": 0.75, "HKG": 0.75},
    ),
    "t10_125": TariffScenario(
        name="t10_125",
        description="10% ROW, 25% on Canada and Mexico, 125% on China and Hong Kong.",
        default_us_tariff=0.10,
        us_import_tariffs={"CAN": 0.25, "MEX": 0.25, "CHN": 1.25, "HKG": 1.25},
    ),
    "t10_145": TariffScenario(
        name="t10_145",
        description="Maximum escalation: 10% ROW, 25% on Canada and Mexico, 145% on China and Hong Kong.",
        default_us_tariff=0.10,
        us_import_tariffs={"CAN": 0.25, "MEX": 0.25, "CHN": 1.45, "HKG": 1.45},
    ),
    # EU 15% retaliation variants
    "t10_25_15eu": TariffScenario(
        name="t10_25_15eu",
        description="EU 15% variant: 10% ROW, 25% NAFTA/China, 15% on EU-27 member states.",
        default_us_tariff=0.10,
        us_import_tariffs={
            "CAN": 0.25,
            "MEX": 0.25,
            "CHN": 0.25,
            "HKG": 0.25,
            **{c: 0.15 for c in EU_COUNTRY_CODES},
        },
    ),
    "t10_54_15eu": TariffScenario(
        name="t10_54_15eu",
        description="EU 15% variant: 10% ROW, 25% NAFTA, 54% China/HKG, 15% on EU-27 member states.",
        default_us_tariff=0.10,
        us_import_tariffs={
            "CAN": 0.25,
            "MEX": 0.25,
            "CHN": 0.54,
            "HKG": 0.54,
            **{c: 0.15 for c in EU_COUNTRY_CODES},
        },
    ),
    "t10_75_15eu": TariffScenario(
        name="t10_75_15eu",
        description="EU 15% variant: 10% ROW, 25% NAFTA, 75% China/HKG, 15% on EU-27 member states.",
        default_us_tariff=0.10,
        us_import_tariffs={
            "CAN": 0.25,
            "MEX": 0.25,
            "CHN": 0.75,
            "HKG": 0.75,
            **{c: 0.15 for c in EU_COUNTRY_CODES},
        },
    ),
    "t10_125_15eu": TariffScenario(
        name="t10_125_15eu",
        description="EU 15% variant: 10% ROW, 25% NAFTA, 125% China/HKG, 15% on EU-27 member states.",
        default_us_tariff=0.10,
        us_import_tariffs={
            "CAN": 0.25,
            "MEX": 0.25,
            "CHN": 1.25,
            "HKG": 1.25,
            **{c: 0.15 for c in EU_COUNTRY_CODES},
        },
    ),
    "t10_145_15eu": TariffScenario(
        name="t10_145_15eu",
        description="EU 15% variant: 10% ROW, 25% NAFTA, 145% China/HKG, 15% on EU-27 member states.",
        default_us_tariff=0.10,
        us_import_tariffs={
            "CAN": 0.25,
            "MEX": 0.25,
            "CHN": 1.45,
            "HKG": 1.45,
            **{c: 0.15 for c in EU_COUNTRY_CODES},
        },
    ),
}

# Aliases
SCENARIOS["15eu"] = SCENARIOS["t10_25_15eu"]
CANONICAL_SCENARIOS = SCENARIOS
CANONICAL_SCENARIO_NAMES: tuple[str, ...] = (
    "base",
    "t10",
    "t10_25",
    "t10_54",
    "t10_75",
    "t10_125",
    "t10_145",
)


def get_canonical_scenario(name: str) -> TariffScenario:
    """Retrieve canonical scenario specification by name."""
    if name not in SCENARIOS:
        valid = list(SCENARIOS.keys())
        raise KeyError(f"Unknown scenario '{name}'. Available canonical scenarios: {valid}")
    return SCENARIOS[name]


def list_canonical_scenarios() -> list[str]:
    """Return list of canonical scenario identifiers in escalation order."""
    return list(CANONICAL_SCENARIO_NAMES)


# ---------------------------------------------------------------------------
# Tariff Matrix Construction
# ---------------------------------------------------------------------------

def build_tariff_matrices(
    scenario: TariffScenario | str | Mapping[str, Any],
    calib: TradeCalibrationResult | None = None,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct 4D bilateral tariff multiplier tensors and national tariff vectors.

    Parameters
    ----------
    scenario : TariffScenario, str, or Mapping
        Scenario specification. If a string, looks up in :data:`SCENARIOS`.
    calib : TradeCalibrationResult
        Calibrated model structural parameters containing dimensions and country codes.

    Returns
    -------
    tau : np.ndarray, shape (11, 77, 11, 77)
        Bilateral intermediate tariff multipliers: ``tau[s_orig, c_orig, s_dest, c_dest] = 1 + rate``.
    tau_fd : np.ndarray, shape (11, 77, 3, 77)
        Bilateral final demand tariff multipliers: ``tau_fd[s_orig, c_orig, ifd_dest, c_dest] = 1 + rate``.
    tauf : np.ndarray, shape (1, 77)
        National uniform intermediate tariff rates (all zeros for bilateral shocks).
    tauf_fd : np.ndarray, shape (1, 77)
        National uniform final demand tariff rates (all zeros for bilateral shocks).
    """
    if isinstance(scenario, TradeCalibrationResult) and not isinstance(calib, TradeCalibrationResult):
        calib, scenario = scenario, calib

    if calib is None:
        raise ValueError("calib: TradeCalibrationResult is required to build tariff matrices.")

    nc = calib.n_countries
    ns = calib.n_sectors
    nfd = calib.n_final_demand
    country_codes = list(calib.country_codes) if calib.country_codes else list(CANONICAL_COUNTRY_CODES)
    sector_codes = list(calib.sector_codes) if calib.sector_codes else list(CANONICAL_SECTOR_CODES)

    # Resolve scenario specification
    if isinstance(scenario, str):
        if scenario in SCENARIOS:
            scen = SCENARIOS[scenario]
        else:
            raise KeyError(f"Unknown scenario '{scenario}'. Available canonical scenarios: {list(SCENARIOS.keys())}")
    elif isinstance(scenario, TariffScenario):
        scen = scenario
    elif isinstance(scenario, Mapping):
        def_tariff = float(scenario.get("default_tariff", kwargs.get("default_rate", 0.0)))
        bi_tariffs = dict(scenario.get("us_import_tariffs", scenario.get("tariffs", {})))
        if not bi_tariffs:
            bi_tariffs = {k: float(v) for k, v in scenario.items() if k in country_codes}
        scen = TariffScenario(
            name=str(scenario.get("name", "custom")),
            description=str(scenario.get("description", "")),
            default_us_tariff=def_tariff,
            us_import_tariffs=bi_tariffs,
        )
    else:
        raise TypeError(f"scenario must be str, TariffScenario, or Mapping, got {type(scenario).__name__}")

    # Validate country codes in scenario
    for c in scen.us_import_tariffs:
        if c not in country_codes:
            raise ValueError(f"Unknown country code '{c}' in scenario us_import_tariffs.")

    for (c_orig, c_dest) in scen.bilateral_tariffs:
        if c_orig not in country_codes:
            raise ValueError(f"Unknown exporter country code '{c_orig}' in bilateral_tariffs.")
        if c_dest not in country_codes:
            raise ValueError(f"Unknown importer country code '{c_dest}' in bilateral_tariffs.")

    for (c_orig, c_dest, s_code) in scen.sectoral_tariffs:
        if c_orig not in country_codes:
            raise ValueError(f"Unknown exporter country code '{c_orig}' in sectoral_tariffs.")
        if c_dest not in country_codes:
            raise ValueError(f"Unknown importer country code '{c_dest}' in sectoral_tariffs.")
        if s_code not in sector_codes:
            raise ValueError(f"Unknown sector code '{s_code}' in sectoral_tariffs.")

    # Initialize all multipliers to 1.0 (0% tariff)
    tau = np.ones((ns, nc, ns, nc), dtype=float)
    tau_fd = np.ones((ns, nc, nfd, nc), dtype=float)
    tauf = np.zeros((1, nc), dtype=float)
    tauf_fd = np.zeros((1, nc), dtype=float)

    # Populate tariffs
    for d_idx, d_code in enumerate(country_codes):
        for o_idx, o_code in enumerate(country_codes):
            # Domestic transactions are never tariffed: tau[s, k, s', k] = 1.0
            if o_idx == d_idx:
                continue

            rate = scen.get_rate(origin_code=o_code, dest_code=d_code)
            if rate < 0.0:
                raise ValueError(f"Tariff rate cannot be negative, got {rate} for {o_code}->{d_code}")

            mult = 1.0 + rate
            tau[:, o_idx, :, d_idx] = mult
            tau_fd[:, o_idx, :, d_idx] = mult

            # Sector overrides
            for s_idx, s_code in enumerate(sector_codes):
                sec_rate = scen.get_rate(origin_code=o_code, dest_code=d_code, sector_code=s_code)
                if sec_rate != rate:
                    sec_mult = 1.0 + sec_rate
                    tau[s_idx, o_idx, :, d_idx] = sec_mult
                    tau_fd[s_idx, o_idx, :, d_idx] = sec_mult

    # Ensure domestic diagonal is strictly 1.0
    for k in range(nc):
        tau[:, k, :, k] = 1.0
        tau_fd[:, k, :, k] = 1.0

    return tau, tau_fd, tauf, tauf_fd


# ---------------------------------------------------------------------------
# Single Scenario Execution
# ---------------------------------------------------------------------------

def run_tariff_scenario(
    scenario: TariffScenario | str | Mapping[str, Any],
    calib: TradeCalibrationResult | None = None,
    *,
    x0: np.ndarray | None = None,
    base_result: TradeEquilibriumResult | None = None,
    method: str = "newton",
    tol: float = 2.5e-3,
    max_iter: int = 50,
    replicate_matlab_precedence: bool = True,
    **kwargs: Any,
) -> TradeEquilibriumResult:
    """Solve general equilibrium for a single tariff counterfactual scenario."""
    if isinstance(scenario, TradeCalibrationResult) and not isinstance(calib, TradeCalibrationResult):
        calib, scenario = scenario, calib

    if calib is None:
        raise ValueError("calib: TradeCalibrationResult is required to run tariff scenario.")

    tau_kwargs = {k: v for k, v in kwargs.items() if k in ("default_rate", "default_tariff")}
    solver_kwargs = {k: v for k, v in kwargs.items() if k not in ("default_rate", "default_tariff")}

    tau, tau_fd, tauf, tauf_fd = build_tariff_matrices(scenario, calib, **tau_kwargs)
    scen_name = scenario if isinstance(scenario, str) else getattr(scenario, "name", "custom")

    res = solve_trade_equilibrium(
        calib=calib,
        tau=tau,
        tau_fd=tau_fd,
        tauf=tauf,
        tauf_fd=tauf_fd,
        x0=x0,
        method=method,
        tol=tol,
        max_iter=max_iter,
        replicate_matlab_precedence=replicate_matlab_precedence,
        base_result=base_result,
        **solver_kwargs,
    )
    res.metadata["scenario_name"] = scen_name
    return res


# ---------------------------------------------------------------------------
# Sequential Batch Execution Engine
# ---------------------------------------------------------------------------

def run_scenario_batch(
    scenarios: Sequence[TariffScenario | str | Mapping[str, Any]] | None = None,
    calib: TradeCalibrationResult | None = None,
    *,
    baseline_scenario: str = "base",
    warm_start: bool = True,
    compute_gk: bool = True,
    matlab_compat: bool = True,
    method: str = "newton",
    tol: float = 2.5e-3,
    max_iter: int = 50,
    replicate_matlab_precedence: bool = True,
    base_result: TradeEquilibriumResult | None = None,
    initial_guesses: dict[str, np.ndarray] | None = None,
    **kwargs: Any,
) -> ScenarioBatchResult:
    """Execute a batch of tariff scenarios using warm-started sequential chaining."""
    if isinstance(scenarios, TradeCalibrationResult) and calib is None:
        calib, scenarios = scenarios, None

    if calib is None:
        raise ValueError("calib: TradeCalibrationResult is required for run_scenario_batch.")

    if scenarios is None:
        scen_list: list[Any] = list(CANONICAL_SCENARIO_NAMES)
    else:
        scen_list = list(scenarios)

    country_codes = tuple(calib.country_codes) if calib.country_codes else tuple(CANONICAL_COUNTRY_CODES)
    sector_codes = tuple(calib.sector_codes) if calib.sector_codes else tuple(CANONICAL_SECTOR_CODES)

    results: dict[str, TradeEquilibriumResult] = {}
    gk_results: dict[str, GearyKhamisResult] = {}
    prev_x: np.ndarray | None = None

    # Step 1: Pre-seed baseline if provided
    if base_result is not None:
        results[baseline_scenario] = base_result
        prev_x = base_result.x_sol.copy()

    # Step 2: Sequential solves
    for s_item in scen_list:
        s_name = s_item if isinstance(s_item, str) else getattr(s_item, "name", "custom")

        # Skip baseline if already provided
        if s_name == baseline_scenario and baseline_scenario in results:
            continue

        if initial_guesses is not None and s_name in initial_guesses:
            x_init = initial_guesses[s_name].copy()
        elif warm_start and prev_x is not None:
            x_init = prev_x.copy()
        else:
            x_init = build_initial_guess(calib)

        eq_res = run_tariff_scenario(
            scenario=s_item,
            calib=calib,
            x0=x_init,
            base_result=results.get(baseline_scenario),
            method=method,
            tol=tol,
            max_iter=max_iter,
            replicate_matlab_precedence=replicate_matlab_precedence,
            **kwargs,
        )
        results[s_name] = eq_res

        if warm_start:
            prev_x = eq_res.x_sol.copy()

        if s_name == baseline_scenario and baseline_scenario not in results:
            results[baseline_scenario] = eq_res

    # Step 3: Geary-Khamis multilateral PPP evaluation
    base_eq = results.get(baseline_scenario)
    if compute_gk and base_eq is not None:
        for s_name, res in results.items():
            gk_res = compute_geary_khamis(
                equilibrium_sol=res,
                base_sol=base_eq,
                calib=calib,
                method="iterative",
                matlab_compat=matlab_compat,
                tol=1e-10,
                max_iter=1000,
            )
            # Baseline real GDP growth is identically 0.0%
            if s_name == baseline_scenario:
                zero_growth = np.zeros(len(gk_res.ppp), dtype=float)
                gk_res = replace(
                    gk_res,
                    gdp_growth=zero_growth,
                    real_gdp_growth=zero_growth,
                )
            gk_results[s_name] = gk_res

    return ScenarioBatchResult(
        scenarios=results,
        geary_khamis=gk_results,
        baseline_scenario=baseline_scenario,
        country_codes=country_codes,
        sector_codes=sector_codes,
        metadata={
            "warm_start": warm_start,
            "method": method,
            "tol": tol,
            "matlab_compat": matlab_compat,
            "replicate_matlab_precedence": replicate_matlab_precedence,
        },
    )


__all__ = [
    "TariffScenario",
    "ExtendedTariffScenario",
    "SCENARIOS",
    "CANONICAL_SCENARIOS",
    "CANONICAL_SCENARIO_NAMES",
    "get_canonical_scenario",
    "list_canonical_scenarios",
    "build_tariff_matrices",
    "run_tariff_scenario",
    "run_scenario_batch",
]
