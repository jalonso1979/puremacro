"""Comparative composite tariff and household distributional policy lab.

Combines :class:`~puremacro.trade.scenarios.CompositeTariffScenario`,
:func:`~puremacro.trade.scenarios.run_scenario_batch`,
:func:`~puremacro.trade.scenarios.summarize_scenario_batch`, and
:func:`~puremacro.trade.distributional.distributional_welfare_from_results`
across comparative trade-policy packages on the bundled OECD ICIO table (or
a caller-supplied :class:`~puremacro.trade._results.TradeCalibrationResult`)
with optional household survey expenditure CSVs, ENIGH 2022 quintiles, or
synthetic income-group benchmarks.

All generated outputs (bilingual ``en``/``es`` Markdown summaries and tables,
CSV exports, PNG figure, JSON record, and SHA-256 ``manifest.json``) avoid
host-specific filesystem paths for reproducible portability.

Run via CLI::

    python -m puremacro.examples.tariff_composite_lab --output ./tariff_lab_out
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from puremacro.examples import _brief_record
from puremacro.examples import first_brief_tariffs as _fbt
from puremacro.study.robust import RobustPolicyComparison
from puremacro.trade._results import (
    ScenarioBatchResult,
    TradeCalibrationResult,
)
from puremacro.trade.data import ICIOData
from puremacro.trade.distributional import (
    DistributionalWelfareResult,
    HouseholdGroups,
    distributional_welfare_from_results,
    prepare_household_groups,
)
from puremacro.trade.scenarios import (
    CompositeTariffScenario,
    ScenarioComparisonResult,
    TariffScenario,
    combine_scenarios,
    run_scenario_batch,
    summarize_scenario_batch,
)

ConflictMode = Literal["last_wins", "first_wins", "error", "max", "sum"]
WelfareRule = Literal["fixed_baskets", "cobb_douglas", "ces"]
HouseholdSource = Literal["auto", "enigh", "synthetic", "csv"]
Language = Literal["en", "es"]

CONFLICT_MODES: tuple[ConflictMode, ...] = (
    "last_wins",
    "first_wins",
    "error",
    "max",
    "sum",
)
WELFARE_RULES: tuple[WelfareRule, ...] = ("fixed_baskets", "cobb_douglas", "ces")
HOUSEHOLD_SOURCES: tuple[HouseholdSource, ...] = ("auto", "enigh", "synthetic", "csv")

SOLVER: dict[str, Any] = dict(_fbt.SOLVER)

_REBATE_RULES: tuple[str, ...] = ("equal", "proportional", "none")

_KNOWN_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "importer",
        "partner",
        "rate",
        "retaliation_rate",
        "exempt_sectors",
        "escalation_sectors",
        "escalation_rate",
        "on_conflict",
        "focus_countries",
        "household_source",
        "groups_path",
        "welfare_rule",
        "ces_elasticity",
        "lang",
        "packages",
        "dataset_label",
        "citation",
    }
)


@dataclass(frozen=True)
class TariffLegSpec:
    """One constituent policy leg within a composite tariff package."""

    importer: str
    tariffs: dict[tuple[str, str], float]
    name: str
    description: str = ""

    def __post_init__(self) -> None:
        imp = str(self.importer).strip()
        if not imp:
            raise ValueError("TariffLegSpec.importer must be a non-empty country code.")
        nm = str(self.name).strip()
        if not nm:
            raise ValueError("TariffLegSpec.name must be a non-empty string.")
        cleaned: dict[tuple[str, str], float] = {}
        for pair, val in self.tariffs.items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError(
                    f"TariffLegSpec.tariffs keys must be (exporter, sector) 2-tuples; got {pair!r}."
                )
            exp, sec = str(pair[0]).strip(), str(pair[1]).strip()
            if not exp or not sec:
                raise ValueError(f"Tariff pair ({exp!r}, {sec!r}) must have non-empty codes.")
            fval = float(val)
            if not math.isfinite(fval) or fval < 0.0 or fval > 10.0:
                raise ValueError(
                    f"Tariff rate for ({exp}, {sec}) must be finite in [0.0, 10.0]; got {val!r}."
                )
            cleaned[(exp, sec)] = fval
        if not cleaned:
            raise ValueError(f"TariffLegSpec {nm!r} must specify at least one tariff entry.")
        object.__setattr__(self, "importer", imp)
        object.__setattr__(self, "name", nm)
        object.__setattr__(self, "tariffs", cleaned)
        object.__setattr__(self, "description", str(self.description))


@dataclass(frozen=True)
class TariffPackageSpec:
    """Declarative specification of a composite tariff policy package."""

    name: str
    legs: tuple[TariffLegSpec, ...]
    on_conflict: ConflictMode = "last_wins"
    description: str = ""

    def __post_init__(self) -> None:
        nm = str(self.name).strip()
        if not nm:
            raise ValueError("TariffPackageSpec.name must be a non-empty string.")
        if not self.legs:
            raise ValueError(f"TariffPackageSpec {nm!r} must include at least one leg.")
        if self.on_conflict not in CONFLICT_MODES:
            raise ValueError(
                f"Unsupported on_conflict={self.on_conflict!r}; expected one of {CONFLICT_MODES}."
            )
        object.__setattr__(self, "name", nm)
        object.__setattr__(self, "legs", tuple(self.legs))
        object.__setattr__(self, "description", str(self.description))


@dataclass(frozen=True)
class TariffLabConfig:
    """Configuration for comparative composite tariff & household welfare simulations."""

    importer: str = "USA"
    partner: str = "CHN"
    rate: float = 0.15
    retaliation_rate: float = 0.15
    exempt_sectors: tuple[str, ...] = ("A", "Q")
    escalation_sectors: tuple[str, ...] = ("C",)
    escalation_rate: float | None = None
    on_conflict: ConflictMode = "last_wins"
    focus_countries: tuple[str, ...] = ("USA", "CHN", "MEX", "DEU", "CAN")
    household_source: HouseholdSource = "auto"
    groups_path: str | None = None
    welfare_rule: WelfareRule = "fixed_baskets"
    ces_elasticity: float = 1.0
    lang: Language = "en"
    packages: tuple[TariffPackageSpec, ...] | None = None
    dataset_label: str = "OECD ICIO (bundled 20x20)"
    citation: str = "OECD Inter-Country Input-Output (ICIO) benchmark + ENIGH 2022 household quintiles"

    def __post_init__(self) -> None:
        imp = str(self.importer).strip()
        prt = str(self.partner).strip()
        if not imp or not prt:
            raise ValueError("importer and partner must be non-empty country codes.")
        if imp == prt and self.packages is None:
            raise ValueError(f"importer and partner must differ when building default packages; got {imp!r}.")

        rate_f = float(self.rate)
        ret_f = float(self.retaliation_rate)
        if not math.isfinite(rate_f) or rate_f <= 0.0 or rate_f > 5.0:
            raise ValueError(f"rate must be finite in (0, 5.0]; got {self.rate!r}.")
        if not math.isfinite(ret_f) or ret_f < 0.0 or ret_f > 5.0:
            raise ValueError(f"retaliation_rate must be finite in [0, 5.0]; got {self.retaliation_rate!r}.")

        esc_rate = float(self.escalation_rate) if self.escalation_rate is not None else min(rate_f + 0.10, 5.0)
        if not math.isfinite(esc_rate) or esc_rate < 0.0 or esc_rate > 5.0:
            raise ValueError(f"escalation_rate must be finite in [0, 5.0]; got {self.escalation_rate!r}.")

        if self.on_conflict not in CONFLICT_MODES:
            raise ValueError(f"Invalid on_conflict={self.on_conflict!r}; expected {CONFLICT_MODES}.")
        if self.welfare_rule not in WELFARE_RULES:
            raise ValueError(f"Invalid welfare_rule={self.welfare_rule!r}; expected {WELFARE_RULES}.")
        if self.household_source not in HOUSEHOLD_SOURCES:
            raise ValueError(
                f"Invalid household_source={self.household_source!r}; expected {HOUSEHOLD_SOURCES}."
            )
        if self.lang not in ("en", "es"):
            raise ValueError(f"Invalid lang={self.lang!r}; expected 'en' or 'es'.")

        ces_sig = float(self.ces_elasticity)
        if not math.isfinite(ces_sig) or ces_sig <= 0.0:
            raise ValueError(f"ces_elasticity must be finite and > 0; got {self.ces_elasticity!r}.")

        gp = str(self.groups_path).strip() if self.groups_path is not None else None
        hh_source = self.household_source
        if gp:
            if hh_source == "auto":
                hh_source = "csv"
        elif hh_source == "csv":
            raise ValueError("household_source='csv' requires groups_path to be set.")

        exempt = tuple(str(s).strip() for s in self.exempt_sectors if str(s).strip())
        esc_secs = tuple(str(s).strip() for s in self.escalation_sectors if str(s).strip())
        focus = tuple(dict.fromkeys([imp, prt, *(str(c).strip() for c in self.focus_countries if str(c).strip())]))

        object.__setattr__(self, "importer", imp)
        object.__setattr__(self, "partner", prt)
        object.__setattr__(self, "rate", rate_f)
        object.__setattr__(self, "retaliation_rate", ret_f)
        object.__setattr__(self, "exempt_sectors", exempt)
        object.__setattr__(self, "escalation_sectors", esc_secs)
        object.__setattr__(self, "escalation_rate", esc_rate)
        object.__setattr__(self, "focus_countries", focus)
        object.__setattr__(self, "household_source", hh_source)
        object.__setattr__(self, "groups_path", gp if gp else None)
        object.__setattr__(self, "ces_elasticity", ces_sig)
        if self.packages is not None:
            pkg_tuple = tuple(self.packages)
            if not pkg_tuple:
                raise ValueError("packages cannot be an empty sequence when explicitly provided.")
            names = [p.name for p in pkg_tuple]
            if len(set(names)) != len(names):
                raise ValueError(f"Duplicate package names are not allowed: {names!r}.")
            if "baseline" in names:
                raise ValueError("Package name 'baseline' is reserved for the benchmark equilibrium.")
            object.__setattr__(self, "packages", pkg_tuple)


@dataclass
class TariffLabResult:
    """Full structured output of a comparative composite-tariff simulation lab."""

    config: TariffLabConfig
    calib: TradeCalibrationResult
    packages: dict[str, TariffScenario | CompositeTariffScenario]
    batch: ScenarioBatchResult
    comparison: ScenarioComparisonResult
    robustness: RobustPolicyComparison
    conflicts: pd.DataFrame
    household_incidence: pd.DataFrame
    distributional_results: dict[tuple[str, str], DistributionalWelfareResult]
    groups_meta: dict[str, Any]
    warnings_en: list[str] = field(default_factory=list)
    warnings_es: list[str] = field(default_factory=list)


_TEXT: dict[Language, dict[str, str]] = {
    "en": {
        "title": "Comparative Composite Tariff & Distributional Welfare Lab",
        "exec": "Executive Summary",
        "packages": "Policy Packages & Composition Conflict Audit",
        "national": "Cross-Country General Equilibrium Comparison",
        "robust": "Minimax Regret & Maximin Robustness Ranking",
        "dist": "Household Distributional Incidence",
        "warnings": "Methodological Notes & Warnings",
        "provenance": "Model & Data Provenance",
        "no_conflicts": "No overlapping bilateral-sector rules triggered across composite packages.",
        "col_welfare": "EV (% cons.)",
        "col_gdp": "Real GDP (%)",
        "col_cpi": "CPI (%)",
        "fig_p1": "Welfare across Countries (EV, % of consumption)",
        "fig_p2": "Macro Effects in Focus Importer",
        "fig_p3": "Household Welfare by Group ({importer})",
        "rebate_equal": "Equal rebate",
        "rebate_prop": "Proportional rebate",
        "rebate_none": "No rebate",
    },
    "es": {
        "title": "Laboratorio Comparativo de Aranceles Compuestos y Bienestar Distributivo",
        "exec": "Resumen Ejecutivo",
        "packages": "Paquetes de Politica y Auditoria de Conflictos de Composicion",
        "national": "Comparacion de Equilibrio General por Pais",
        "robust": "Ordenamiento por Robustez Maximin y Minimax Arrepentimiento",
        "dist": "Incidencia Distributiva en los Hogares",
        "warnings": "Notas Metodologicas y Advertencias",
        "provenance": "Trazabilidad de Modelo y Datos",
        "no_conflicts": "No se detectaron traslapes sector-origen entre reglas compuestas.",
        "col_welfare": "VE (% cons.)",
        "col_gdp": "PIB real (%)",
        "col_cpi": "IPC (%)",
        "fig_p1": "Bienestar por Pais (VE, % del consumo)",
        "fig_p2": "Efectos Macro en el Importador Focal",
        "fig_p3": "Bienestar por Grupo de Hogares ({importer})",
        "rebate_equal": "Reembolso uniforme",
        "rebate_prop": "Reembolso proporcional",
        "rebate_none": "Sin reembolso",
    },
}


def _parse_leg_dict(raw: Mapping[str, Any]) -> TariffLegSpec:
    if not isinstance(raw, Mapping):
        raise ValueError("Each package leg in JSON config must be an object.")
    raw_tariffs = raw.get("tariffs")
    parsed_tariffs: dict[tuple[str, str], float] = {}
    if isinstance(raw_tariffs, Sequence) and not isinstance(raw_tariffs, (str, bytes)):
        for entry in raw_tariffs:
            if not isinstance(entry, Mapping):
                raise ValueError(f"List tariffs entries must be objects; got {entry!r}.")
            parsed_tariffs[(str(entry["exporter"]), str(entry["sector"]))] = float(entry["rate"])
    elif isinstance(raw_tariffs, Mapping):
        for k, v in raw_tariffs.items():
            if ":" in str(k):
                exp, sec = str(k).split(":", 1)
            elif "," in str(k):
                exp, sec = str(k).split(",", 1)
            else:
                raise ValueError(
                    f"Tariff mapping key {k!r} must use 'EXPORTER:SECTOR' or list syntax."
                )
            parsed_tariffs[(exp.strip(), sec.strip())] = float(v)
    else:
        raise ValueError("Leg 'tariffs' must be a list of {exporter, sector, rate} or an object.")
    return TariffLegSpec(
        importer=str(raw.get("importer", "")),
        tariffs=parsed_tariffs,
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
    )


def load_config(path: str | Path) -> TariffLabConfig:
    """Load and validate a :class:`TariffLabConfig` from a UTF-8 JSON file."""
    cfg_path = Path(path)
    raw = json.loads(cfg_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"Config JSON in {cfg_path.name} must be a JSON object.")
    unknown = set(raw.keys()) - _KNOWN_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys in {cfg_path.name}: {sorted(unknown)}")

    kwargs: dict[str, Any] = dict(raw)
    if "exempt_sectors" in kwargs and kwargs["exempt_sectors"] is not None:
        kwargs["exempt_sectors"] = tuple(str(x) for x in kwargs["exempt_sectors"])
    if "escalation_sectors" in kwargs and kwargs["escalation_sectors"] is not None:
        kwargs["escalation_sectors"] = tuple(str(x) for x in kwargs["escalation_sectors"])
    if "focus_countries" in kwargs and kwargs["focus_countries"] is not None:
        kwargs["focus_countries"] = tuple(str(x) for x in kwargs["focus_countries"])
    if "groups_path" in kwargs and kwargs["groups_path"] is not None:
        gp_raw = Path(str(kwargs["groups_path"]))
        if not gp_raw.is_absolute():
            gp_raw = (cfg_path.parent / gp_raw).resolve()
        kwargs["groups_path"] = str(gp_raw)
    if "packages" in kwargs and kwargs["packages"] is not None:
        raw_pkgs = kwargs["packages"]
        if not isinstance(raw_pkgs, Sequence) or isinstance(raw_pkgs, (str, bytes)):
            raise ValueError("'packages' in config JSON must be a list of package definitions.")
        parsed_pkgs: list[TariffPackageSpec] = []
        for p in raw_pkgs:
            if not isinstance(p, Mapping):
                raise ValueError("Each package entry must be a JSON object.")
            raw_legs = p.get("legs")
            if not isinstance(raw_legs, Sequence) or isinstance(raw_legs, (str, bytes)):
                raise ValueError(f"Package {p.get('name', '?')!r} must provide a list of 'legs'.")
            legs = tuple(_parse_leg_dict(leg) for leg in raw_legs)
            parsed_pkgs.append(
                TariffPackageSpec(
                    name=str(p.get("name", "")),
                    legs=legs,
                    on_conflict=p.get("on_conflict", "last_wins"),
                    description=str(p.get("description", "")),
                )
            )
        kwargs["packages"] = tuple(parsed_pkgs)
    return TariffLabConfig(**kwargs)


def default_tariff_packages(config: TariffLabConfig) -> tuple[TariffPackageSpec, ...]:
    """Build comparative unilateral, retaliatory, and phased-exempted packages."""
    imp = config.importer
    prt = config.partner
    r = config.rate
    ret = config.retaliation_rate
    esc_rate = config.escalation_rate if config.escalation_rate is not None else min(r + 0.10, 5.0)

    unilateral_leg = TariffLegSpec(
        importer=imp,
        tariffs={(prt, "*"): r},
        name=f"{imp}_unilateral",
        description=f"{imp} places {r:.1%} across-the-board tariff on {prt}.",
    )
    retaliation_leg = TariffLegSpec(
        importer=prt,
        tariffs={(imp, "*"): ret},
        name=f"{prt}_retaliation",
        description=f"{prt} places {ret:.1%} retaliatory tariff on {imp}.",
    )
    escalation_map: dict[tuple[str, str], float] = {
        (prt, sec): esc_rate for sec in config.escalation_sectors
    }
    exempt_map: dict[tuple[str, str], float] = {
        (prt, sec): 0.0 for sec in config.exempt_sectors
    }

    phased_legs: list[TariffLegSpec] = [unilateral_leg]
    if escalation_map:
        phased_legs.append(
            TariffLegSpec(
                importer=imp,
                tariffs=escalation_map,
                name="targeted_escalation",
                description=f"Stage-2 escalation ({esc_rate:.1%}) on strategic sectors.",
            )
        )
    if ret > 0.0:
        phased_legs.append(retaliation_leg)
    if exempt_map:
        phased_legs.append(
            TariffLegSpec(
                importer=imp,
                tariffs=exempt_map,
                name="strategic_carveout",
                description="Explicit 0.0% carve-out on exempted sectors.",
            )
        )

    return (
        TariffPackageSpec(
            name="unilateral",
            legs=(unilateral_leg,),
            on_conflict="last_wins",
            description=f"Unilateral {imp} tariff on {prt} ({r:.1%}).",
        ),
        TariffPackageSpec(
            name="retaliatory",
            legs=(unilateral_leg, retaliation_leg),
            on_conflict=config.on_conflict,
            description=f"Bilateral escalation: {imp} ({r:.1%}) + {prt} retaliation ({ret:.1%}).",
        ),
        TariffPackageSpec(
            name="phased_exempted",
            legs=tuple(phased_legs),
            on_conflict=config.on_conflict,
            description="Phased escalation with bilateral retaliation and zero-rate sector carve-outs.",
        ),
    )


def build_tariff_packages(
    config: TariffLabConfig,
    calib: TradeCalibrationResult,
) -> tuple[dict[str, CompositeTariffScenario], pd.DataFrame]:
    """Validate codes against ``calib`` and build composite scenarios plus conflict audit."""
    country_set = set(calib.country_codes)
    sector_set = set(calib.sector_codes)
    specs = config.packages if config.packages is not None else default_tariff_packages(config)

    if config.importer not in country_set:
        raise ValueError(
            f"Unknown importer {config.importer!r}; available countries: {list(calib.country_codes)}."
        )
    if config.packages is None and config.partner not in country_set:
        raise ValueError(
            f"Unknown partner {config.partner!r}; available countries: {list(calib.country_codes)}."
        )
    for country in config.focus_countries:
        if country not in country_set:
            raise ValueError(
                f"Unknown focus country {country!r}; available countries: {list(calib.country_codes)}."
            )

    packages: dict[str, CompositeTariffScenario] = {}
    conflict_frames: list[pd.DataFrame] = []

    for pkg in specs:
        scen_legs: list[TariffScenario] = []
        for leg in pkg.legs:
            if leg.importer not in country_set:
                raise ValueError(
                    f"Leg {leg.name!r} targets unknown importer {leg.importer!r}; "
                    f"expected one of {list(calib.country_codes)}."
                )
            for exp, sec in leg.tariffs:
                if exp != "*" and exp not in country_set:
                    raise ValueError(
                        f"Leg {leg.name!r} specifies unknown exporter {exp!r}."
                    )
                if sec != "*" and sec not in sector_set:
                    raise ValueError(
                        f"Leg {leg.name!r} specifies unknown sector {sec!r}; "
                        f"expected '*' or one of {list(calib.sector_codes)}."
                    )
            scen_legs.append(
                TariffScenario(
                    importer=leg.importer,
                    tariffs=leg.tariffs,
                    name=leg.name,
                    description=leg.description,
                )
            )
        composite = combine_scenarios(
            scen_legs,
            name=pkg.name,
            on_conflict=pkg.on_conflict,
            description=pkg.description,
        )
        # Trigger tensor materialization early so on_conflict='error' raises immediately.
        _ = composite.to_rate_tensor(calib.country_codes, calib.sector_codes)
        cf = composite.conflicts(calib.country_codes, calib.sector_codes)
        if not cf.empty:
            cf = cf.copy()
            cf.insert(0, "package", pkg.name)
            conflict_frames.append(cf)
        packages[pkg.name] = composite

    if conflict_frames:
        conflicts_df = pd.concat(conflict_frames, ignore_index=True)
    else:
        conflicts_df = pd.DataFrame(
            columns=[
                "package",
                "exporter",
                "importer",
                "sector",
                "n_sources",
                "sources",
                "resolved_rate",
                "rule",
            ]
        )
    return packages, conflicts_df


def _validate_group_frame(df: pd.DataFrame, sector_codes: Sequence[str]) -> pd.DataFrame:
    sectors = list(sector_codes)
    required = ["population_weight", "annual_spending", *sectors]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Household groups table missing required columns: {missing}")
    if df.empty:
        raise ValueError("Household groups table must contain at least one group.")

    out = df[required].copy()
    if out.index.has_duplicates:
        raise ValueError("Household groups index must not contain duplicate group labels.")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="raise")
        if not np.all(np.isfinite(out[col].to_numpy(dtype=float))):
            raise ValueError(f"Household groups column {col!r} contains non-finite values.")
    if np.any(out["population_weight"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("All population_weight entries must be strictly positive.")
    if np.any(out["annual_spending"].to_numpy(dtype=float) <= 0.0):
        raise ValueError("All annual_spending entries must be strictly positive.")
    shares = out[sectors].to_numpy(dtype=float)
    if np.any(shares < 0.0):
        raise ValueError("Sector spending shares cannot be negative.")
    row_sums = shares.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(
            f"Each household group's sector shares must sum to 1.0 within 1e-6; got {row_sums.tolist()}."
        )
    return out


def load_lab_groups(
    config: TariffLabConfig,
    calib: TradeCalibrationResult,
    *,
    icio: ICIOData | None = None,
    groups_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], list[str], list[str]]:
    """Load and validate household groups together with portable provenance metadata."""
    warnings_en: list[str] = []
    warnings_es: list[str] = []
    sectors = list(calib.sector_codes)

    if groups_df is not None:
        clean = _validate_group_frame(groups_df, sectors)
        meta: dict[str, Any] = {
            "source": "supplied_dataframe",
            "importer": config.importer,
            "monetary_unit": "currency/year",
            "period": "benchmark",
            "file": None,
            "sha256": None,
        }
        return clean, meta, warnings_en, warnings_es

    if config.household_source == "csv" or config.groups_path is not None:
        assert config.groups_path is not None
        gp = Path(config.groups_path)
        if not gp.is_file():
            raise FileNotFoundError(f"Household groups CSV not found: {gp.name}")
        raw_df = pd.read_csv(gp, encoding="utf-8")
        if "group" not in raw_df.columns:
            raise ValueError(f"Household CSV {gp.name} must contain a 'group' column.")
        expected = {"group", "population_weight", "annual_spending", *sectors}
        extra = sorted(set(raw_df.columns) - expected)
        if extra:
            raise ValueError(f"Unexpected columns in {gp.name}: {extra}")
        raw_df = raw_df.set_index("group")
        clean = _validate_group_frame(raw_df, sectors)
        meta = {
            "source": "user_csv",
            "importer": config.importer,
            "monetary_unit": "local_currency/year",
            "period": "user_survey",
            "file": gp.name,
            "sha256": _brief_record.sha256(gp),
        }
        return clean, meta, warnings_en, warnings_es

    if icio is None:
        if config.household_source in ("enigh", "synthetic"):
            raise ValueError(
                "Bundled ENIGH / synthetic ICIO group builders require the ICIOData object."
            )
        # Fallback synthetic equal-sector groups sized cleanly on calib final demand
        idx = list(calib.country_codes).index(config.importer)
        fd_orig = np.asarray(calib.afd, dtype=float)[:, 0, idx].reshape(
            calib.n_countries, calib.n_sectors
        ).sum(axis=0)
        shares = np.where(fd_orig > 0.0, 1.0, 0.0)
        if shares.sum() <= 0.0:
            raise ValueError(f"Calibration has no final-demand support for {config.importer!r}.")
        shares = shares / shares.sum()
        rows = {
            sec: [float(shares[i])] * 3 for i, sec in enumerate(sectors)
        }
        syn_df = pd.DataFrame(
            {
                "population_weight": [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                "annual_spending": [30000.0, 60000.0, 120000.0],
                **rows,
            },
            index=pd.Index(["low", "middle", "high"], name="group"),
        )
        warnings_en.append(
            f"No survey CSV provided; using stylized equal-share tertiles for {config.importer}."
        )
        warnings_es.append(
            f"Sin encuesta CSV; se usan terciles estilizados de participacion uniforme para {config.importer}."
        )
        meta = {
            "source": "stylized_tertiles",
            "importer": config.importer,
            "monetary_unit": "currency/year",
            "period": "synthetic",
            "file": None,
            "sha256": None,
        }
        return syn_df, meta, warnings_en, warnings_es

    use_enigh = config.household_source == "enigh" or (
        config.household_source == "auto" and config.importer == "MEX"
    )
    if use_enigh:
        enigh_df = _fbt.load_enigh_groups(icio, config.importer)
        clean = _validate_group_frame(enigh_df, sectors)
        if config.importer != "MEX":
            warnings_en.append(
                f"ENIGH 2022 expenditure quintiles are empirical Mexico survey shares mapped onto {config.importer} GE prices."
            )
            warnings_es.append(
                f"Los quintiles ENIGH 2022 provienen de Mexico y se proyectan sobre precios de equilibrio general de {config.importer}."
            )
        meta = {
            "source": "enigh_2022",
            "importer": config.importer,
            "monetary_unit": "MXN/year",
            "period": "2022",
            "file": "ENIGH_2022_bundled",
            "sha256": None,
        }
        return clean, meta, warnings_en, warnings_es

    syn_df = _fbt.synthetic_groups(icio, config.importer)
    clean = _validate_group_frame(syn_df, sectors)
    warnings_en.append(
        f"Using stylized three-group synthetic expenditure distribution matched to {config.importer} OECD ICIO final demand; supply --groups CSV for empirical national microdata."
    )
    warnings_es.append(
        f"Se utiliza una distribucion sintetica de tres grupos calibrada a la demanda final OECD ICIO de {config.importer}; proporcione --groups CSV para microdatos nacionales."
    )
    meta = {
        "source": "synthetic_tertiles",
        "importer": config.importer,
        "monetary_unit": "USD/year",
        "period": "2018",
        "file": None,
        "sha256": None,
    }
    return clean, meta, warnings_en, warnings_es


def _compute_distributional_table(
    config: TariffLabConfig,
    calib: TradeCalibrationResult,
    batch: ScenarioBatchResult,
    groups_df: pd.DataFrame,
    groups_meta: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[tuple[str, str], DistributionalWelfareResult]]:
    importer = config.importer
    idx = list(calib.country_codes).index(importer)
    base_res = batch.scenarios[batch.baseline_scenario]
    sectors = list(calib.sector_codes)

    w0 = float(np.asarray(base_res.w_sol, dtype=float).ravel()[idx])
    r0 = float(np.asarray(base_res.r_sol, dtype=float).ravel()[idx])
    T0 = float(np.asarray(base_res.T_sol, dtype=float).ravel()[idx])
    L = float(np.asarray(calib.l_endow, dtype=float).ravel()[idx])
    K = float(np.asarray(calib.k_endow, dtype=float).ravel()[idx])

    labor0 = w0 * L
    cap0 = r0 * K
    tot0 = labor0 + cap0 + T0
    if tot0 <= 0.0 or w0 <= 0.0 or r0 <= 0.0:
        raise ValueError(f"Non-positive baseline factor income or expenditure for {importer!r}.")

    expenditure_df = groups_df[sectors].multiply(groups_df["annual_spending"], axis=0)
    income_exposure = pd.DataFrame(
        {
            "labor": groups_df["annual_spending"] * (labor0 / tot0),
            "capital": groups_df["annual_spending"] * (cap0 / tot0),
        },
        index=groups_df.index,
    )
    hh_groups: HouseholdGroups = prepare_household_groups(
        expenditure_df,
        groups_df["population_weight"],
        monetary_unit=str(groups_meta.get("monetary_unit", "currency/year")),
        period=str(groups_meta.get("period", "benchmark")),
        provenance={"source": groups_meta.get("source", "unknown"), "importer": importer},
        income_exposure=income_exposure,
        baseline_budget=groups_df["annual_spending"],
    )

    agg_spend = float((groups_df["annual_spending"] * groups_df["population_weight"]).sum())
    extra_rule_kwargs: dict[str, Any] = {"rule": config.welfare_rule}
    if config.welfare_rule == "ces":
        extra_rule_kwargs["ces_elasticity"] = config.ces_elasticity

    rows: list[pd.DataFrame] = []
    dw_map: dict[tuple[str, str], DistributionalWelfareResult] = {}

    for scen_name, cf_res in batch.scenarios.items():
        if scen_name == batch.baseline_scenario:
            continue
        w1 = float(np.asarray(cf_res.w_sol, dtype=float).ravel()[idx])
        r1 = float(np.asarray(cf_res.r_sol, dtype=float).ravel()[idx])
        T1 = float(np.asarray(cf_res.T_sol, dtype=float).ravel()[idx])
        fic = pd.Series({"labor": w1 / w0 - 1.0, "capital": r1 / r0 - 1.0})
        rebate_total = agg_spend * ((T1 - T0) / tot0)

        for rebate_rule in _REBATE_RULES:
            if rebate_rule == "equal":
                t_tot = rebate_total
                t_weights = None
            elif rebate_rule == "proportional":
                t_tot = rebate_total
                t_weights = groups_df["annual_spending"]
            else:
                t_tot = 0.0
                t_weights = None

            dw = distributional_welfare_from_results(
                hh_groups,
                calib,
                base_res,
                cf_res,
                country=importer,
                factor_income_changes=fic,
                transfer_total=t_tot,
                transfer_weights=t_weights,
                **extra_rule_kwargs,
            )
            dw_map[(scen_name, rebate_rule)] = dw
            gdf = dw.groups.copy()
            gdf = gdf.reset_index().rename(columns={"index": "group"})
            if "group" not in gdf.columns:
                gdf.insert(0, "group", [str(ix) for ix in dw.groups.index])
            gdf.insert(0, "rebate_rule", rebate_rule)
            gdf.insert(0, "scenario", scen_name)
            gdf["importer"] = importer
            rows.append(gdf)

    combined = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return combined, dw_map


def compute_tariff_lab(
    config: TariffLabConfig,
    *,
    calib: TradeCalibrationResult | None = None,
    icio: ICIOData | None = None,
    groups_df: pd.DataFrame | None = None,
    solver_opts: Mapping[str, Any] | None = None,
) -> TariffLabResult:
    """Solve all comparative tariff packages and household distributional welfare."""
    loaded_icio: ICIOData | None = icio
    loaded_calib: TradeCalibrationResult
    if calib is None:
        loaded_icio, loaded_calib = _fbt.load_bundled()
    else:
        loaded_calib = calib

    packages, conflicts_df = build_tariff_packages(config, loaded_calib)
    hh_df, groups_meta, warn_en, warn_es = load_lab_groups(
        config, loaded_calib, icio=loaded_icio, groups_df=groups_df
    )

    opts = dict(SOLVER)
    if solver_opts:
        opts.update(dict(solver_opts))

    batch = run_scenario_batch(
        loaded_calib,
        list(packages.values()),
        baseline_name="baseline",
        **opts,
    )
    non_conv = [k for k, v in batch.scenarios.items() if not v.converged]
    if non_conv:
        warn_en.append(f"Scenarios without full Newton convergence: {non_conv}.")
        warn_es.append(f"Escenarios sin convergencia completa de Newton: {non_conv}.")

    comparison = summarize_scenario_batch(
        batch,
        loaded_calib,
        countries=config.focus_countries,
    )
    robustness = comparison.compare_robust(metric="ev_pct_consumption")
    incidence_df, dw_map = _compute_distributional_table(
        config, loaded_calib, batch, hh_df, groups_meta
    )

    return TariffLabResult(
        config=config,
        calib=loaded_calib,
        packages=dict(packages),
        batch=batch,
        comparison=comparison,
        robustness=robustness,
        conflicts=conflicts_df,
        household_incidence=incidence_df,
        distributional_results=dw_map,
        groups_meta=groups_meta,
        warnings_en=warn_en,
        warnings_es=warn_es,
    )


def _df_to_markdown(df: pd.DataFrame, digits: int = 3) -> str:
    if df.empty:
        return "_(empty)_"
    fmt_df = df.copy()
    for col in fmt_df.columns:
        if pd.api.types.is_bool_dtype(fmt_df[col]) or pd.api.types.is_integer_dtype(fmt_df[col]):
            continue
        if pd.api.types.is_numeric_dtype(fmt_df[col]):
            fmt_df[col] = fmt_df[col].map(
                lambda x: f"{float(x):.{digits}f}" if pd.notna(x) else ""
            )
    cols = [str(c) for c in fmt_df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_rows = [
        "| " + " | ".join(str(row[c]) for c in fmt_df.columns) + " |"
        for _, row in fmt_df.iterrows()
    ]
    return "\n".join([header, sep, *body_rows])


def render_summary(result: TariffLabResult, lang: Language = "en") -> str:
    """Render a concise bilingual summary in English (``'en'``) or Spanish (``'es'``)."""
    if lang not in ("en", "es"):
        raise ValueError(f"Unsupported language {lang!r}; expected 'en' or 'es'.")
    t = _TEXT[lang]
    cfg = result.config
    summary_df = result.comparison.to_frame("summary")
    compact_df = result.comparison.to_frame("compact")
    rob_df = result.robustness.summary.reset_index()

    focus_cf = compact_df[
        (compact_df["country"] == cfg.importer) & (compact_df["scenario"] != "baseline")
    ].copy()

    lines: list[str] = [f"# {t['title']}", ""]
    lines.append(f"## {t['exec']}")
    lines.append("")
    if lang == "en":
        lines.append(
            f"- **Focus importer**: `{cfg.importer}` | **Dataset**: {cfg.dataset_label}"
        )
        lines.append(
            f"- **Household survey source**: `{result.groups_meta['source']}` "
            f"({result.groups_meta['period']}, `{result.groups_meta['monetary_unit']}`)"
        )
        lines.append(f"- **Preference engine**: `{cfg.welfare_rule}`")
        lines.append(
            f"- **Packages evaluated**: {', '.join(result.packages.keys())} "
            f"(overlap conflict rows: {len(result.conflicts)})"
        )
    else:
        lines.append(
            f"- **Importador focal**: `{cfg.importer}` | **Conjunto de datos**: {cfg.dataset_label}"
        )
        lines.append(
            f"- **Fuente de hogares**: `{result.groups_meta['source']}` "
            f"({result.groups_meta['period']}, `{result.groups_meta['monetary_unit']}`)"
        )
        lines.append(f"- **Regla de preferencias**: `{cfg.welfare_rule}`")
        lines.append(
            f"- **Paquetes evaluados**: {', '.join(result.packages.keys())} "
            f"(conflictos resueltos: {len(result.conflicts)})"
        )
    lines.append("")

    if not focus_cf.empty:
        lines.append(f"### {t['national']} (`{cfg.importer}`)")
        lines.append("")
        lines.append(_df_to_markdown(focus_cf))
        lines.append("")

    lines.append(f"## {t['packages']}")
    lines.append("")
    if result.conflicts.empty:
        lines.append(t["no_conflicts"])
    else:
        lines.append(_df_to_markdown(result.conflicts))
    lines.append("")

    lines.append(f"## {t['robust']}")
    lines.append("")
    lines.append(_df_to_markdown(rob_df))
    lines.append("")

    lines.append(f"## {t['dist']} (`{cfg.importer}`)")
    lines.append("")
    dist_cols = [
        "scenario",
        "rebate_rule",
        "group",
        "ev_pct",
        "cv_pct",
        "price_ev",
        "income_ev",
        "transfer_ev",
    ]
    dist_view = result.household_incidence[dist_cols] if not result.household_incidence.empty else result.household_incidence
    lines.append(_df_to_markdown(dist_view))
    lines.append("")

    lines.append("## " + ("Scenario Batch Summary" if lang == "en" else "Resumen por Paquete"))
    lines.append("")
    lines.append(_df_to_markdown(summary_df))
    lines.append("")

    warns = result.warnings_en if lang == "en" else result.warnings_es
    if warns:
        lines.append(f"## {t['warnings']}")
        lines.append("")
        for w in warns:
            lines.append(f"- {w}")
        lines.append("")

    lines.append(f"## {t['provenance']}")
    lines.append("")
    lines.append(f"- **Citation**: {cfg.citation}")
    lines.append("")
    return "\n".join(lines)


def render_tables_markdown(result: TariffLabResult, lang: Language = "en") -> str:
    """Render full comparative policy tables in Markdown."""
    if lang not in ("en", "es"):
        raise ValueError(f"Unsupported language {lang!r}; expected 'en' or 'es'.")
    t = _TEXT[lang]
    sections = [
        f"# {t['title']} — " + ("Tables" if lang == "en" else "Tablas"),
        "",
        f"## {t['national']}",
        "",
        result.comparison.to_markdown("compact", digits=3),
        "",
        "## " + ("Payoff Matrix (EV %)" if lang == "en" else "Matriz de Pagos (VE %)"),
        "",
        _df_to_markdown(result.comparison.to_payoff_matrix().reset_index()),
        "",
        f"## {t['robust']}",
        "",
        _df_to_markdown(result.robustness.summary.reset_index()),
        "",
        f"## {t['packages']}",
        "",
        _df_to_markdown(result.conflicts) if not result.conflicts.empty else t["no_conflicts"],
        "",
        f"## {t['dist']}",
        "",
        _df_to_markdown(result.household_incidence),
        "",
    ]
    return "\n".join(sections)


def render_comparison_figure(result: TariffLabResult, path: Path, *, lang: Language = "en") -> None:
    """Render a 3-panel comparative figure using matplotlib's non-interactive Agg backend."""
    if lang not in ("en", "es"):
        raise ValueError(f"Unsupported language {lang!r}; expected 'en' or 'es'.")
    t = _TEXT[lang]
    cfg = result.config

    fig = Figure(figsize=(14, 4.6), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 3)

    # Panel 1: Country EV payoff heatmap/grouped bars
    payoff = result.comparison.to_payoff_matrix(include_baseline=False)
    ax0 = axes[0]
    if not payoff.empty:
        x = np.arange(len(payoff.columns), dtype=float)
        n_scen = len(payoff.index)
        width = 0.78 / max(n_scen, 1)
        for i, scen_name in enumerate(payoff.index):
            offset = (i - (n_scen - 1) / 2.0) * width
            ax0.bar(x + offset, payoff.loc[scen_name].to_numpy(dtype=float), width=width, label=str(scen_name))
        ax0.set_xticks(x)
        ax0.set_xticklabels([str(c) for c in payoff.columns], fontsize=8)
        ax0.axhline(0.0, color="#333333", linewidth=0.8)
        ax0.legend(fontsize=7, frameon=False)
    ax0.set_title(t["fig_p1"], fontsize=9.5)
    ax0.set_ylabel(t["col_welfare"], fontsize=8.5)

    # Panel 2: Real GDP vs CPI in focus importer
    compact = result.comparison.to_frame("compact")
    imp_df = compact[(compact["country"] == cfg.importer) & (compact["scenario"] != "baseline")]
    ax1 = axes[1]
    if not imp_df.empty:
        x1 = np.arange(len(imp_df), dtype=float)
        w1 = 0.35
        ax1.bar(x1 - w1 / 2.0, imp_df["real_gdp_growth_pct"].to_numpy(dtype=float), width=w1, label=t["col_gdp"])
        ax1.bar(x1 + w1 / 2.0, imp_df["cpi_inflation_pct"].to_numpy(dtype=float), width=w1, label=t["col_cpi"])
        ax1.set_xticks(x1)
        ax1.set_xticklabels([str(s) for s in imp_df["scenario"]], fontsize=8, rotation=15)
        ax1.axhline(0.0, color="#333333", linewidth=0.8)
        ax1.legend(fontsize=7, frameon=False)
    ax1.set_title(f"{t['fig_p2']} ({cfg.importer})", fontsize=9.5)

    # Panel 3: Household distributional EV (%) across groups (last counterfactual or all packages)
    ax2 = axes[2]
    inc = result.household_incidence
    if not inc.empty:
        # Compare final/headline counterfactual across rebate rules for clearest distributional comparison
        target_scen = str(inc["scenario"].iloc[-1])
        sub = inc[(inc["scenario"] == target_scen) & (inc["rebate_rule"].isin(["equal", "proportional", "none"]))]
        groups_order = list(dict.fromkeys(sub["group"].astype(str).tolist()))
        rules = [r for r in ("equal", "proportional", "none") if r in set(sub["rebate_rule"])]
        x2 = np.arange(len(groups_order), dtype=float)
        w2 = 0.78 / max(len(rules), 1)
        rule_labels = {
            "equal": t["rebate_equal"],
            "proportional": t["rebate_prop"],
            "none": t["rebate_none"],
        }
        for i, rr in enumerate(rules):
            rsub = sub[sub["rebate_rule"] == rr].set_index("group")
            vals = [float(rsub.loc[g, "ev_pct"]) for g in groups_order]
            offset = (i - (len(rules) - 1) / 2.0) * w2
            ax2.bar(x2 + offset, vals, width=w2, label=rule_labels.get(rr, rr))
        ax2.set_xticks(x2)
        ax2.set_xticklabels(groups_order, fontsize=8)
        ax2.axhline(0.0, color="#333333", linewidth=0.8)
        ax2.legend(fontsize=7, frameon=False)
        ax2.set_title(f"{t['fig_p3'].format(importer=cfg.importer)} [{target_scen}]", fontsize=9.5)
    else:
        ax2.set_title(t["fig_p3"].format(importer=cfg.importer), fontsize=9.5)
    ax2.set_ylabel(t["col_welfare"], fontsize=8.5)

    fig.tight_layout()
    fig.savefig(path, dpi=150, format="png")
    fig.clear()


def _sanitized_config_dict(config: TariffLabConfig, groups_meta: Mapping[str, Any]) -> dict[str, Any]:
    raw = asdict(config)
    raw["groups_path"] = groups_meta.get("file") if config.groups_path is not None else None
    raw["exempt_sectors"] = list(config.exempt_sectors)
    raw["escalation_sectors"] = list(config.escalation_sectors)
    raw["focus_countries"] = list(config.focus_countries)
    if config.packages is not None:
        raw["packages"] = [
            {
                "name": p.name,
                "on_conflict": p.on_conflict,
                "description": p.description,
                "legs": [
                    {
                        "importer": leg.importer,
                        "name": leg.name,
                        "description": leg.description,
                        "tariffs": [
                            {"exporter": k[0], "sector": k[1], "rate": v}
                            for k, v in sorted(leg.tariffs.items())
                        ],
                    }
                    for leg in p.legs
                ],
            }
            for p in config.packages
        ]
    return raw


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    clean = df.replace({np.nan: None})
    records: list[dict[str, Any]] = []
    for row in clean.to_dict(orient="records"):
        fixed: dict[str, Any] = {}
        for k, v in row.items():
            if isinstance(v, (np.floating, np.integer)):
                fixed[str(k)] = v.item()
            elif isinstance(v, np.bool_):
                fixed[str(k)] = bool(v)
            elif isinstance(v, tuple):
                fixed[str(k)] = list(v)
            else:
                fixed[str(k)] = v
        records.append(fixed)
    return records


def build_lab_record(result: TariffLabResult) -> dict[str, Any]:
    """Build a portable JSON-serializable record with zero host paths."""
    cfg = result.config
    return {
        "schema": 1,
        "module": "puremacro.examples.tariff_composite_lab",
        "config": _sanitized_config_dict(cfg, result.groups_meta),
        "provenance": {
            "dataset_label": cfg.dataset_label,
            "citation": cfg.citation,
            "groups": dict(result.groups_meta),
        },
        "solver": dict(SOLVER),
        "packages": list(result.packages.keys()),
        "conflicts": _json_safe_records(result.conflicts),
        "scenario_summary": _json_safe_records(result.comparison.to_frame("summary")),
        "national_comparison": _json_safe_records(result.comparison.to_frame("compact")),
        "robustness": _json_safe_records(result.robustness.summary.reset_index()),
        "household_incidence": _json_safe_records(result.household_incidence),
        "warnings": {"en": list(result.warnings_en), "es": list(result.warnings_es)},
    }


def write_outputs(result: TariffLabResult, output_dir: str | Path) -> dict[str, Path]:
    """Write bilingual summaries, Markdown/CSV/JSON tables, PNG chart, and SHA-256 manifest."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    lang = result.config.lang

    summary_en = render_summary(result, lang="en")
    summary_es = render_summary(result, lang="es")
    summary_primary = summary_en if lang == "en" else summary_es

    written: dict[str, Path] = {}

    def _write_text(rel: str, text: str) -> Path:
        p = out / rel
        p.write_text(text, encoding="utf-8")
        written[rel] = p
        return p

    def _write_csv(rel: str, df: pd.DataFrame) -> Path:
        p = out / rel
        df.to_csv(p, index=False, encoding="utf-8")
        written[rel] = p
        return p

    _write_text("summary.md", summary_primary)
    _write_text("summary_en.md", summary_en)
    _write_text("summary_es.md", summary_es)
    _write_text("tables_en.md", render_tables_markdown(result, lang="en"))
    _write_text("tables_es.md", render_tables_markdown(result, lang="es"))

    _write_csv("national_comparison.csv", result.comparison.to_frame("full"))
    _write_csv("scenario_summary.csv", result.comparison.to_frame("summary"))
    _write_csv("payoff_matrix.csv", result.comparison.to_payoff_matrix().reset_index())
    _write_csv("robustness_summary.csv", result.robustness.summary.reset_index())
    _write_csv("tariff_conflicts.csv", result.conflicts)
    _write_csv("household_incidence.csv", result.household_incidence)

    fig_path = out / "comparison.png"
    render_comparison_figure(result, fig_path, lang=lang)
    written["comparison.png"] = fig_path

    rec_path = out / "record.json"
    _brief_record.write_json_atomic(rec_path, build_lab_record(result))
    written["record.json"] = rec_path

    artifacts = {
        rel: {
            "sha256": _brief_record.sha256(p),
            "bytes": int(p.stat().st_size),
        }
        for rel, p in sorted(written.items())
    }
    manifest = {
        "schema": 1,
        "module": "puremacro.examples.tariff_composite_lab",
        "lang": lang,
        "environment": _brief_record.environment(),
        "inputs": {
            "dataset_label": result.config.dataset_label,
            "groups_source": result.groups_meta.get("source"),
            "groups_file": result.groups_meta.get("file"),
            "groups_sha256": result.groups_meta.get("sha256"),
        },
        "artifacts": artifacts,
    }
    manifest_path = out / "manifest.json"
    _brief_record.write_json_atomic(manifest_path, manifest)
    written["manifest.json"] = manifest_path
    return written


def run_tariff_lab(
    config: TariffLabConfig,
    output_dir: str | Path,
    *,
    calib: TradeCalibrationResult | None = None,
    icio: ICIOData | None = None,
    groups_df: pd.DataFrame | None = None,
    solver_opts: Mapping[str, Any] | None = None,
) -> tuple[TariffLabResult, dict[str, Path]]:
    """Run comparative composite-tariff policy lab end-to-end and export all outputs."""
    result = compute_tariff_lab(
        config,
        calib=calib,
        icio=icio,
        groups_df=groups_df,
        solver_opts=solver_opts,
    )
    paths = write_outputs(result, output_dir)
    return result, paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.examples.tariff_composite_lab",
        description="Comparative composite-tariff & household distributional policy lab.",
    )
    parser.add_argument("--config", type=Path, help="JSON config file overriding TariffLabConfig.")
    parser.add_argument("--output", type=Path, help="Directory to export tables, summaries, figure, and manifest.")
    parser.add_argument("--lang", choices=("en", "es"), help="Primary output language ('en' or 'es').")
    parser.add_argument("--importer", type=str, help="Target importer ISO3 code (default: USA).")
    parser.add_argument("--partner", type=str, help="Primary bilateral partner ISO3 code (default: CHN).")
    parser.add_argument("--rate", type=float, help="Unilateral tariff rate as decimal (e.g. 0.15).")
    parser.add_argument("--retaliation-rate", type=float, help="Partner retaliatory tariff rate (default: 0.15).")
    parser.add_argument("--exempt-sectors", type=str, help="Comma-separated sectors for 0%% carve-outs (e.g. A,Q).")
    parser.add_argument("--on-conflict", choices=CONFLICT_MODES, help="Composite overlap resolution rule.")
    parser.add_argument("--countries", type=str, help="Comma-separated focus countries (e.g. USA,CHN,MEX,DEU).")
    parser.add_argument("--groups", type=Path, help="Optional CSV file with household spending shares.")
    parser.add_argument("--enigh", action="store_true", help="Force bundled ENIGH 2022 household quintiles.")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic household income groups.")
    parser.add_argument("--welfare-rule", choices=WELFARE_RULES, help="Household welfare preference rule.")
    return parser


def _build_cli_config(args: argparse.Namespace) -> TariffLabConfig:
    base = load_config(args.config) if args.config is not None else TariffLabConfig()
    overrides = asdict(base)
    if args.lang is not None:
        overrides["lang"] = args.lang
    if args.importer is not None:
        overrides["importer"] = args.importer
    if args.partner is not None:
        overrides["partner"] = args.partner
    if args.rate is not None:
        overrides["rate"] = float(args.rate)
    if args.retaliation_rate is not None:
        overrides["retaliation_rate"] = float(args.retaliation_rate)
    if args.exempt_sectors is not None:
        overrides["exempt_sectors"] = tuple(
            s.strip() for s in args.exempt_sectors.split(",") if s.strip()
        )
    if args.on_conflict is not None:
        overrides["on_conflict"] = args.on_conflict
    if args.countries is not None:
        overrides["focus_countries"] = tuple(
            s.strip() for s in args.countries.split(",") if s.strip()
        )
    if args.groups is not None:
        overrides["groups_path"] = str(args.groups)
        overrides["household_source"] = "csv"
    elif args.enigh and args.synthetic:
        raise ValueError("Cannot pass both --enigh and --synthetic.")
    elif args.enigh:
        overrides["household_source"] = "enigh"
    elif args.synthetic:
        overrides["household_source"] = "synthetic"
    if args.welfare_rule is not None:
        overrides["welfare_rule"] = args.welfare_rule

    # Reconstitute nested package dataclasses cleanly if inherited from load_config
    if base.packages is not None:
        overrides["packages"] = base.packages
    return TariffLabConfig(**overrides)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for the composite tariff policy lab."""
    parser = _build_parser()
    arg_list = list(argv) if argv is not None else None
    args = parser.parse_args(arg_list)
    if arg_list is not None and not arg_list:
        parser.print_help()
        return 0
    if args.output is None:
        parser.print_help()
        return 0
    config = _build_cli_config(args)
    result, paths = run_tariff_lab(config, args.output)
    print(f"Wrote {len(paths)} artifacts for {len(result.packages)} composite packages.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
