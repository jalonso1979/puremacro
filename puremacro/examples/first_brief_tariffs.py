"""Your first policy brief: a 10% US tariff across all modeled Chinese sectors on the bundled OECD ICIO 2019 table, national EV with its price / factor-income / fiscal split, and synthetic household groups.

Run it with::

    python -m puremacro.examples.first_brief_tariffs --output first_brief

The brief solves the exact trade engine (``calibrate_trade_model`` +
``solve_trade_equilibrium``) twice on the bundled OECD Inter-Country
Input-Output table (2023 edition, reference year 2019, 19 G20 economies plus
ROW, 20 ISIC Rev.4 sections): once without tariffs and once with the
scenario. It reports the importer's Hicksian equivalent variation (EV) with
its endpoint split into purchaser prices, factor income and fiscal transfers
(``compute_hicksian_welfare``), maps the same GE prices and incomes to
household groups (``distributional_welfare_from_results``) and, on request,
repeats the solve under one alternative assumption (``--experiment
sigma=2``: substitution across all country-sector intermediate inputs).

The default household groups are SYNTHETIC, method illustration only: three
groups built from the importer's national consumption basket by documented
tilts. Supply your own group spending shares with ``--groups CSV``.

The experiment is unilateral, with no retaliation: China does not respond.
The three EV lines are an endpoint attribution valued in the model's
numeraire, not separate causal channels; the record also reports each
country's terms of trade and the importer's wage relative to the exporter's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import textwrap
import time
from collections.abc import Collection, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import puremacro
from puremacro.reports import df_to_latex, df_to_markdown
from puremacro.trade import (
    TariffScenario,
    calibrate_trade_model,
    compute_hicksian_welfare,
    run_scenario,
    solve_trade_equilibrium,
)
from puremacro.trade.data import ICIOData, load_icio_data
from puremacro.trade.distributional import (
    distributional_welfare_from_results,
    prepare_household_groups,
)
from ._brief_record import environment, sha256 as _record_sha256, write_json_atomic

__all__ = [
    "SOLVER", "SYNTHETIC_LABEL", "SYNTHETIC_LABEL_ES", "DEFAULT_EXPERIMENT_SIGMA",
    "BriefSolution", "TradeBriefConfig", "load_bundled", "validate_scenario", "solve_brief",
    "national_record", "parse_experiment", "national_basket", "supported_sectors",
    "synthetic_groups", "read_groups_csv", "load_enigh_groups", "load_groups",
    "household_incidence", "compute_sensitivity_grid", "compute_brief", "headline_record",
    "page_block", "summary_text", "write_outputs", "run_brief", "main",
]

#: Solver settings of every state (baseline, counterfactual, experiment).
#: ``world_income`` makes EV invariant to the order of countries in the table.
SOLVER: dict[str, Any] = {"method": "equilibrated_newton", "tol": 1e-5,
                          "foreign_saving_units": "world_income"}
SYNTHETIC_LABEL = "SYNTHETIC, method illustration only"
SYNTHETIC_LABEL_ES = "SINTÉTICO, solo ilustración del método (SYNTHETIC, method illustration only)"
DEFAULT_EXPERIMENT_SIGMA = 2.0

#: Synthetic groups: mean spending (USD per household per year) and the tilt
#: ``d`` toward agriculture (A) and manufactures incl. food (C). The high
#: group's tilt is set so the spending-weighted mean basket is the national one.
SYNTHETIC_SPENDING = {"low": 30_000.0, "middle": 60_000.0, "high": 120_000.0}
SYNTHETIC_TILT = {"low": 0.30, "middle": 0.0, "high": -0.30 * 30_000.0 / 120_000.0}
GOODS_SECTORS = ("A", "C")
_GROUP_COLUMNS = ("population_weight", "annual_spending")


@dataclass
class BriefSolution:
    """Everything one brief solve produced; reused by the household and output stages."""

    icio: ICIOData
    calib: Any
    base: Any
    counterfactual: Any
    welfare: Any
    scenario: TariffScenario
    settings: dict[str, Any]
    timings: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeBriefConfig:
    """Explicit configuration for running the trade policy brief on bundled or user data."""

    scenario: str = "USA_tariff_on_CHN_all"
    tariff_rate: float = 0.10
    trade_elasticity: float = 4.0
    household_elasticity: float = 1.0
    use_enigh: bool = True
    data_path: str | Path | None = None
    output_dir: str | Path | None = None
    bilingual: bool = True
    importer: str = "USA"
    exporter: str = "CHN"
    sector: str = "all"
    sensitivity_trade_elasticities: tuple[float, ...] = (2.0, 4.0, 6.0)
    sensitivity_household_elasticities: tuple[float, ...] = (0.5, 1.0, 1.5)

    def __post_init__(self) -> None:
        if not math.isfinite(self.tariff_rate) or self.tariff_rate < 0:
            raise ValueError(f"tariff_rate must be a finite non-negative rate, got {self.tariff_rate!r}")
        if not math.isfinite(self.trade_elasticity) or self.trade_elasticity < 0:
            raise ValueError(f"trade_elasticity must be finite and non-negative, got {self.trade_elasticity!r}")
        if not math.isfinite(self.household_elasticity) or self.household_elasticity < 0:
            raise ValueError(f"household_elasticity must be finite and non-negative, got {self.household_elasticity!r}")

        # If scenario was supplied and differs from default or has _tariff_on_, validate and sync
        if self.scenario and "_tariff_on_" in self.scenario:
            parts = self.scenario.split("_tariff_on_")
            imp = parts[0]
            rest = parts[1].split("_")
            exp = rest[0]
            sec = rest[1] if len(rest) > 1 else "all"
            if self.importer == "USA" and imp != "USA":
                object.__setattr__(self, "importer", imp)
            if self.exporter == "CHN" and exp != "CHN":
                object.__setattr__(self, "exporter", exp)
            if self.sector == "all" and sec != "all":
                object.__setattr__(self, "sector", sec)
        elif self.importer != "USA" or self.exporter != "CHN" or self.sector != "all":
            object.__setattr__(self, "scenario", f"{self.importer}_tariff_on_{self.exporter}_{self.sector}")

        if self.data_path is not None:
            p = Path(self.data_path)
            if not p.is_file():
                raise FileNotFoundError(f"data_path {str(p)!r} does not exist or is not a file")
            object.__setattr__(self, "data_path", p)
        if self.output_dir is not None:
            object.__setattr__(self, "output_dir", Path(self.output_dir))

        object.__setattr__(self, "sensitivity_trade_elasticities",
                           tuple(float(s) for s in self.sensitivity_trade_elasticities))
        object.__setattr__(self, "sensitivity_household_elasticities",
                           tuple(float(h) for h in self.sensitivity_household_elasticities))
        if len(self.sensitivity_trade_elasticities) == 0 or any(s <= 0 for s in self.sensitivity_trade_elasticities):
            raise ValueError(f"sensitivity_trade_elasticities must be positive numbers, got {self.sensitivity_trade_elasticities}")
        if len(self.sensitivity_household_elasticities) == 0 or any(h < 0 for h in self.sensitivity_household_elasticities):
            raise ValueError(f"sensitivity_household_elasticities must be non-negative numbers, got {self.sensitivity_household_elasticities}")


def _load_table() -> ICIOData:
    table = load_icio_data(return_structured=True)
    if not isinstance(table, ICIOData):  # return_structured=True always gives ICIOData
        raise TypeError("load_icio_data(return_structured=True) did not return ICIOData")
    return table


def load_bundled() -> tuple[ICIOData, Any]:
    """The bundled OECD ICIO 2019 table and its calibration."""
    icio = _load_table()
    calib = calibrate_trade_model(icio.matrix, country_codes=icio.country_codes,
                                  sector_codes=icio.sector_codes)
    return icio, calib


def validate_scenario(icio: ICIOData, importer: str, exporter: str, sector: str,
                      tariff: float) -> None:
    """Raise ``ValueError`` naming the offending value of a scenario, before any solve."""
    countries = list(icio.country_codes)
    if importer not in countries:
        raise ValueError(f"unknown importer {importer!r}; known codes: {', '.join(countries)}")
    if exporter not in countries:
        raise ValueError(f"unknown exporter {exporter!r}; known codes: {', '.join(countries)}")
    if exporter == importer:
        raise ValueError(f"exporter {exporter!r} equals the importer; domestic flows are never tariffed")
    if sector != "all" and sector not in icio.sector_codes:
        raise ValueError(f"unknown sector {sector!r}; use 'all' or one of {', '.join(icio.sector_codes)}")
    try:
        value = float(tariff)
    except (TypeError, ValueError):
        raise ValueError(f"tariff {tariff!r} is not a number") from None
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"tariff {value!r} must be a finite non-negative rate (0.10 = 10%)")


def parse_experiment(text: str) -> float | None:
    """``"none"`` -> None; ``"sigma=<float>"`` -> that positive finite sigma."""
    text = str(text).strip()
    if text.lower() == "none":
        return None
    name, sep, value = text.partition("=")
    if name.strip() != "sigma" or not sep:
        raise ValueError(f"experiment {text!r} must be 'none' or 'sigma=<value>' "
                         "(the exact engine has no trade elasticity theta)")
    try:
        sigma = float(value)
    except ValueError:
        raise ValueError(f"experiment {text!r}: sigma {value!r} is not a number") from None
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"experiment {text!r}: sigma must be finite and positive "
                         "(sigma=0 is the headline, Leontief sourcing)")
    return sigma


def _require_converged(state: Any, name: str) -> None:
    if not getattr(state, "converged", False):
        termination = getattr(state, "metadata", {}).get("solver_termination")
        raise RuntimeError(f"{name} did not converge (solver_termination={termination!r})")


def solve_brief(importer: str = "USA", exporter: str = "CHN", sector: str = "all",
                tariff: float = 0.10, *, sigma: float = 0.0,
                bundled: tuple[ICIOData, Any] | None = None) -> BriefSolution:
    """Solve baseline and counterfactual on the bundled table and the importer's EV."""
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if bundled is None:
        icio = _load_table()
        timings["load"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        calib = calibrate_trade_model(icio.matrix, country_codes=icio.country_codes,
                                      sector_codes=icio.sector_codes)
        timings["calibration"] = time.perf_counter() - t0
    else:
        icio, calib = bundled
        timings["load"] = 0.0
        timings["calibration"] = 0.0
    validate_scenario(icio, importer, exporter, sector, tariff)
    scenario = TariffScenario(importer, {(exporter, "*" if sector == "all" else sector): float(tariff)},
                              name=f"{importer}_tariff_on_{exporter}_{sector}")
    t0 = time.perf_counter()
    base = solve_trade_equilibrium(calib, sigma=sigma, **SOLVER)
    timings["baseline"] = time.perf_counter() - t0
    _require_converged(base, "baseline")
    t0 = time.perf_counter()
    counterfactual = run_scenario(calib, scenario, base_result=base, sigma=sigma, **SOLVER)
    timings["counterfactual"] = time.perf_counter() - t0
    _require_converged(counterfactual, "counterfactual")
    t0 = time.perf_counter()
    welfare = compute_hicksian_welfare(calib, counterfactual, base_result=base,
                                       target_country=importer)
    timings["welfare"] = time.perf_counter() - t0
    settings = {"importer": importer, "exporter": exporter, "sector": sector,
                "tariff": float(tariff), "sigma": float(sigma), **SOLVER}
    return BriefSolution(icio, calib, base, counterfactual, welfare, scenario, settings, timings)


def _pct_change(base: Any, counterfactual: Any, index: int) -> float:
    return 100.0 * (float(np.ravel(counterfactual)[index]) / float(np.ravel(base)[index]) - 1.0)


def national_record(sol: BriefSolution) -> dict[str, float]:
    """The importer's EV and its split, in million USD (the table's unit) and percent.

    Also the terms of trade (export over import unit values, before duties) of
    importer and exporter, their wages in the model's numeraire, and the
    importer's wage relative to the exporter's (numeraire-free), as % changes.
    """
    w = sol.welfare
    countries = list(sol.calib.country_codes)
    i, e = countries.index(sol.settings["importer"]), countries.index(sol.settings["exporter"])
    b, c = sol.base, sol.counterfactual
    wage_i, wage_e = _pct_change(b.w_sol, c.w_sol, i), _pct_change(b.w_sol, c.w_sol, e)
    # C is the welfare basket; I and Cx are separate quantity changes valued
    # at baseline purchaser prices, not additional consumption EV components.
    nf = sol.calib.n_final_demand
    final_change = (np.asarray(b.Pfd_final) * (np.asarray(c.c_sol) - np.asarray(b.c_sol)))
    final_change = final_change.ravel(order="F").reshape(len(countries), nf)[i]
    categories = list(sol.icio.fd_codes)
    return {
        "ev_musd": float(w.ev),
        "cv_musd": float(w.cv),
        "ev_pct_consumption": float(w.ev_pct_consumption),
        "ev_pct_gdp": float(w.ev_pct_gdp),
        "consumption_base_musd": float(w.consumption_base),
        "price_effect_musd": float(w.price_effect),
        "factor_income_effect_musd": float(w.factor_income_effect),
        "fiscal_transfer_effect_musd": float(w.fiscal_transfer_effect),
        "tariff_rebate_effect_musd": float(w.tariff_rebate_effect),
        "decomposition_residual_musd": float(w.decomposition_residual),
        "terms_of_trade_pct_importer": _pct_change(b.terms_of_trade, c.terms_of_trade, i),
        "terms_of_trade_pct_exporter": _pct_change(b.terms_of_trade, c.terms_of_trade, e),
        "wage_pct_importer": wage_i,
        "wage_pct_exporter": wage_e,
        "relative_wage_pct": 100.0 * ((1.0 + wage_i / 100.0) / (1.0 + wage_e / 100.0) - 1.0),
        "investment_change_base_prices_musd": float(final_change[categories.index("I")]),
        "abroad_change_base_prices_musd": float(final_change[categories.index("Cx")]),
    }


# --- household groups -------------------------------------------------------

def _consumption_by_sector(icio: ICIOData, importer: str) -> np.ndarray:
    countries, sectors, fds = icio.country_codes, icio.sector_codes, icio.fd_codes
    if importer not in countries:
        raise ValueError(f"unknown importer {importer!r}")
    nc, ns, nfd = len(countries), len(sectors), len(fds)
    n = nc * ns
    column = n + countries.index(importer) * nfd + fds.index("C")
    return np.asarray(icio.matrix[:n, column], dtype=float).reshape(nc, ns).sum(axis=0)


def national_basket(icio: ICIOData, importer: str) -> pd.Series:
    """The importer's final consumption (category C) by sector over all origins, as shares."""
    spend = _consumption_by_sector(icio, importer)
    return pd.Series(spend / spend.sum(), index=list(icio.sector_codes), name="share")


def supported_sectors(icio: ICIOData, importer: str) -> list[str]:
    """Sectors the importer's consumption basket buys (positive C purchases)."""
    spend = _consumption_by_sector(icio, importer)
    return [code for code, value in zip(icio.sector_codes, spend) if value > 0]


def synthetic_groups(icio: ICIOData, importer: str) -> pd.DataFrame:
    """Three SYNTHETIC household groups built from the national basket by documented tilts.

    ``s_g = s * (1 + d_g z)`` with ``s`` the national basket and
    ``z = 1[sector in {A, C}] - sum_{A,C} s``, so every row sums to one and the
    spending-weighted mean basket is ``s``. Low spenders tilt toward
    agriculture and manufactures (including food), high spenders toward
    services. Population weights are 1/3 each (a population of size one).
    """
    s = national_basket(icio, importer)
    goods = s.index.isin(GOODS_SECTORS).astype(float)
    z = goods - float(s[s.index.isin(GOODS_SECTORS)].sum())
    rows = {}
    for group, spending in SYNTHETIC_SPENDING.items():
        shares = s.to_numpy() * (1.0 + SYNTHETIC_TILT[group] * z)
        rows[group] = {"population_weight": 1.0 / 3.0, "annual_spending": spending,
                       **dict(zip(s.index, shares))}
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "group"
    frame.attrs = {"source": "synthetic groups from the bundled national basket "
                             "(puremacro.examples.first_brief_tariffs)",
                   "is_synthetic": True, "label": SYNTHETIC_LABEL}
    return frame


def read_groups_csv(path: str | Path, sector_codes: Sequence[str],
                    supported: Collection[str]) -> pd.DataFrame:
    """Read user household groups: ``group, population_weight, annual_spending`` and one
    budget-share column per sector code (each >= 0, row sum 1 within 1e-6), nothing else."""
    path = Path(path)
    raw = pd.read_csv(path, dtype={"group": str})
    sectors = [str(c) for c in sector_codes]
    expected = ["group", *_GROUP_COLUMNS, *sectors]
    missing = [c for c in expected if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")
    extra = [str(c) for c in raw.columns if c not in expected]
    if extra:
        raise ValueError(f"{path.name}: unknown columns {extra}; allowed are "
                         "group, population_weight, annual_spending and the sector codes")
    if raw.empty:
        raise ValueError(f"{path.name}: no groups")
    groups = raw["group"]
    if groups.isna().any() or (groups.str.strip() == "").any():
        raise ValueError(f"{path.name}: every row needs a nonempty 'group' label")
    duplicated = sorted(set(groups[groups.duplicated()]))
    if duplicated:
        raise ValueError(f"{path.name}: duplicate group {duplicated[0]!r}")
    frame = raw.set_index("group")[[*_GROUP_COLUMNS, *sectors]]
    try:
        frame = frame.astype(float)
    except (TypeError, ValueError):
        raise ValueError(f"{path.name}: every value except 'group' must be numeric") from None
    allowed = set(supported)
    for group, row in frame.iterrows():
        if not np.isfinite(row.to_numpy()).all():
            bad = [c for c in frame.columns if not np.isfinite(row[c])]
            raise ValueError(f"{path.name}: group {group!r} has non-finite {bad}")
        for column in _GROUP_COLUMNS:
            if row[column] <= 0:
                raise ValueError(f"{path.name}: group {group!r} needs positive {column}, got {row[column]!r}")
        negative = [s for s in sectors if row[s] < 0]
        if negative:
            raise ValueError(f"{path.name}: group {group!r} has negative shares in sectors {negative}")
        total = float(row[sectors].sum())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{path.name}: group {group!r} shares sum to {total:.6g}, not 1")
        unsupported = [s for s in sectors if row[s] > 0 and s not in allowed]
        if unsupported:
            raise ValueError(f"{path.name}: group {group!r} spends on sectors {unsupported} that the "
                             "importer's consumption basket does not buy in the table")
    frame.index.name = "group"
    frame.attrs = {"source": path.name, "file": path.name, "is_synthetic": False, "label": "user-supplied groups"}
    return frame


def _enigh_path() -> tuple[Path, dict[str, Any]]:
    folder = Path(__file__).resolve().parents[1] / "trade" / "_datafiles"
    path = folder / "enigh2022_household_policy.npz"
    manifest_path = folder / "enigh2022_household_policy.json"
    if not path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"bundled ENIGH 2022 dataset not found in {folder}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload = path.read_bytes()
    actual_hash = hashlib.sha256(payload).hexdigest()
    if actual_hash != manifest.get("payload_sha256"):
        raise ValueError(f"ENIGH input checksum mismatch: expected {manifest.get('payload_sha256')}, got {actual_hash}")
    return path, manifest


def load_enigh_groups(icio: ICIOData, importer: str = "USA") -> pd.DataFrame:
    """Load empirical household quintiles from the bundled INEGI ENIGH 2022 dataset.

    The 90,033 analysis households from ENIGH 2022 are ordered by per-capita current
    income (ing_cor / tot_integ) using household expansion factors (factor) and partitioned
    into five population-weighted quintiles (Q1..Q5). Within each quintile, food expenditure
    (alimentos) and other monetary spending (gasto_mon - alimentos) define the Engel curve,
    mapped to the importer's ICIO consumption basket: food to Agriculture (A) and Food/Manufacturing (C),
    and other spending to remaining sectors proportionally to national consumption shares.
    """
    path, manifest = _enigh_path()
    with np.load(path, allow_pickle=False) as f:
        data = pd.DataFrame({k: f[k] for k in manifest["variables"]})
    data = data.loc[data["gasto_mon"] > 0].copy()
    w = data["factor"].to_numpy(dtype=float)
    rank_income = (data["ing_cor"] / data["tot_integ"]).to_numpy(dtype=float)
    order = np.argsort(rank_income, kind="stable")
    cdf = np.cumsum(w[order]) / w.sum()
    cutoffs = rank_income[order][np.searchsorted(cdf, [0.2, 0.4, 0.6, 0.8])]
    data["quintile"] = np.searchsorted(cutoffs, rank_income, side="left") + 1

    s_nat = national_basket(icio, importer)
    sectors = list(icio.sector_codes)
    goods_mask = s_nat.index.isin(GOODS_SECTORS)
    s_goods = float(s_nat[goods_mask].sum())
    s_services = float(s_nat[~goods_mask].sum())
    if s_goods <= 0:
        s_goods = 1.0
    if s_services <= 0:
        s_services = 1.0

    total_w = float(w.sum())
    rows = {}
    for q in range(1, 6):
        sub = data.loc[data["quintile"] == q]
        sub_w = sub["factor"].to_numpy(dtype=float)
        pop_wt = float(sub_w.sum()) / total_w
        spend_quarterly = float((sub_w * sub["gasto_mon"]).sum()) / float(sub_w.sum())
        annual_spend = float(spend_quarterly * 4.0)
        food_spend = float((sub_w * sub["alimentos"]).sum())
        tot_spend = float((sub_w * sub["gasto_mon"]).sum())
        food_share = food_spend / tot_spend if tot_spend > 0 else 0.4
        non_food_share = 1.0 - food_share

        q_shares = {}
        for sec in sectors:
            if sec in GOODS_SECTORS:
                q_shares[sec] = food_share * float(s_nat[sec]) / s_goods
            else:
                q_shares[sec] = non_food_share * float(s_nat[sec]) / s_services
        tot_s = sum(q_shares.values())
        if tot_s > 0:
            q_shares = {sec: val / tot_s for sec, val in q_shares.items()}

        rows[f"Q{q}"] = {
            "population_weight": pop_wt,
            "annual_spending": annual_spend,
            **q_shares,
        }

    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "group"
    frame.attrs = {
        "source": "INEGI ENIGH 2022 nueva serie (concentradohogar)",
        "file": path.name,
        "sha256": manifest["payload_sha256"],
        "is_synthetic": False,
        "label": "ENIGH 2022 income quintiles (INEGI microdata)",
    }
    return frame


def load_groups(data_path: str | Path, icio: ICIOData, importer: str = "USA") -> pd.DataFrame:
    """Load household groups from user-supplied CSV file (group shares or household microdata)."""
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"household groups file not found: {path}")
    raw = pd.read_csv(path)
    sectors = list(icio.sector_codes)
    supported = supported_sectors(icio, importer)

    # Case 1: Group shares CSV
    if "group" in raw.columns and "population_weight" in raw.columns and "annual_spending" in raw.columns:
        return read_groups_csv(path, sectors, supported)

    # Case 2: Household microdata CSV
    cols = {c.lower(): c for c in raw.columns}
    wt_col = cols.get("weight") or cols.get("factor") or cols.get("pop_weight")
    spend_col = cols.get("spending") or cols.get("gasto_mon") or cols.get("expenditure")
    inc_col = cols.get("income") or cols.get("ing_cor")
    food_col = cols.get("food") or cols.get("alimentos")
    hh_size_col = cols.get("tot_integ") or cols.get("hh_size") or cols.get("members")

    if wt_col and spend_col:
        raw = raw.loc[raw[spend_col] > 0].copy()
        if raw.empty:
            raise ValueError(f"{path.name}: no positive expenditure records found in {spend_col!r} column")
        w = raw[wt_col].to_numpy(dtype=float)
        if not np.isfinite(w).all() or (w <= 0).any():
            raise ValueError(f"{path.name}: weights column {wt_col!r} must contain finite, strictly positive values")
        if len(raw) < 5:
            raise ValueError(
                f"{path.name}: microdata must contain at least 5 observations to construct quintiles, got {len(raw)}"
            )
        hh_size = raw[hh_size_col].to_numpy(dtype=float) if hh_size_col else np.ones(len(raw))
        inc = raw[inc_col].to_numpy(dtype=float) if inc_col else raw[spend_col].to_numpy(dtype=float)
        rank_income = inc / np.maximum(hh_size, 1.0)
        order = np.argsort(rank_income, kind="stable")
        cdf = np.cumsum(w[order]) / w.sum()
        cutoffs = rank_income[order][np.searchsorted(cdf, [0.2, 0.4, 0.6, 0.8])]
        raw["quintile"] = np.searchsorted(cutoffs, rank_income, side="left") + 1

        s_nat = national_basket(icio, importer)
        goods_mask = s_nat.index.isin(GOODS_SECTORS)
        s_goods = float(s_nat[goods_mask].sum()) or 1.0
        s_services = float(s_nat[~goods_mask].sum()) or 1.0
        total_w = float(w.sum())
        rows = {}
        for q in range(1, 6):
            sub = raw.loc[raw["quintile"] == q]
            sub_w = sub[wt_col].to_numpy(dtype=float)
            if sub.empty or float(sub_w.sum()) <= 0:
                raise ValueError(f"{path.name}: quintile Q{q} has no valid observations or zero weight")
            pop_wt = float(sub_w.sum()) / total_w
            mean_spend = float((sub_w * sub[spend_col]).sum()) / float(sub_w.sum())
            if food_col:
                food_spend = float((sub_w * sub[food_col]).sum())
                tot_spend = float((sub_w * sub[spend_col]).sum())
                food_share = food_spend / tot_spend if tot_spend > 0 else 0.4
            else:
                food_share = max(0.1, 0.45 - 0.05 * (q - 1))
            non_food_share = 1.0 - food_share
            q_shares = {}
            for sec in sectors:
                if sec in GOODS_SECTORS:
                    q_shares[sec] = food_share * float(s_nat[sec]) / s_goods
                else:
                    q_shares[sec] = non_food_share * float(s_nat[sec]) / s_services
            tot_s = sum(q_shares.values())
            q_shares = {sec: val / tot_s for sec, val in q_shares.items()}
            rows[f"Q{q}"] = {"population_weight": pop_wt, "annual_spending": mean_spend, **q_shares}
        frame = pd.DataFrame.from_dict(rows, orient="index")
        frame.index.name = "group"
        frame.attrs = {
            "source": path.name,
            "file": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "is_synthetic": False,
            "label": f"User microdata quintiles ({path.name})",
        }
        return frame

    raise ValueError(
        f"{path.name}: unrecognized columns {list(raw.columns)}. Must be either group shares "
        f"(group, population_weight, annual_spending, [sectors]) or household microdata "
        f"with weight and spending columns."
    )


def _importer_incomes(state: Any, calib: Any, index: int) -> tuple[float, float]:
    factor = (float(np.ravel(state.w_sol)[index]) * float(np.ravel(calib.l_endow)[index])
              + float(np.ravel(state.r_sol)[index]) * float(np.ravel(calib.k_endow)[index]))
    return factor, float(np.ravel(state.T_sol)[index])


def household_incidence(sol: BriefSolution, groups: pd.DataFrame,
                        household_elasticity: float = 0.0) -> pd.DataFrame:
    """Group EV (% of group spending) from the brief's GE prices and incomes.

    Each group's budget is split into factor income and transfers in the
    importer's national proportions ``(wL+rK)/(wL+rK+T)``; factor income moves
    with the national ``wL+rK`` (w and r of a country move together in this
    engine) and the change in national transfers ``T`` is rebated either in
    proportion to baseline spending (reproduces the national EV) or equally
    per household.
    """
    started = time.perf_counter()
    importer = sol.settings["importer"]
    sectors = list(sol.icio.sector_codes)
    index = list(sol.calib.country_codes).index(importer)
    f0, t0 = _importer_incomes(sol.base, sol.calib, index)
    f1, t1 = _importer_incomes(sol.counterfactual, sol.calib, index)
    weights = groups["population_weight"].astype(float)
    spending = groups["annual_spending"].astype(float)
    expenditure = groups[sectors].astype(float).mul(spending, axis=0)
    synthetic = bool(groups.attrs.get("is_synthetic", False))
    exposure = pd.DataFrame({"factor_income": spending * f0 / (f0 + t0)})
    prepared = prepare_household_groups(
        expenditure, weights, monetary_unit="USD" if synthetic else "currency units",
        period="year", income_exposure=exposure, provenance=dict(groups.attrs))
    pool = (t1 - t0) / (f0 + t0) * float((weights * spending).sum())
    changes = pd.Series({"factor_income": f1 / f0 - 1.0})
    results = {}
    rule = "ces" if household_elasticity > 0 else "fixed_baskets"
    kw: dict[str, Any] = {"ces_elasticity": float(household_elasticity)} if household_elasticity > 0 else {}
    for r_name, scores in (("proportional", spending), ("equal", None)):
        results[r_name] = distributional_welfare_from_results(
            prepared, sol.calib, sol.base, sol.counterfactual, country=importer,
            factor_income_changes=changes, transfer_total=pool, transfer_weights=scores,
            rule=rule, **kw)
    prop, equal = results["proportional"].groups, results["equal"].groups
    budget = prop["baseline_budget"]
    out = pd.DataFrame({
        "population_weight": weights.to_numpy(),
        "annual_spending": spending.to_numpy(),
        "ev_pct_proportional": prop["ev_pct"].to_numpy(),
        "ev_pct_equal": equal["ev_pct"].to_numpy(),
        "price_ev_pct": (100.0 * prop["price_ev"] / budget).to_numpy(),
        "income_ev_pct": (100.0 * prop["income_ev"] / budget).to_numpy(),
    }, index=groups.index.copy())
    out.index.name = "group"
    out.attrs = {
        **groups.attrs,
        "elapsed_seconds": time.perf_counter() - started,
        "aggregate_ev_pct_proportional": float(results["proportional"].aggregate["ev_pct"]),
        "aggregate_ev_pct_equal": float(results["equal"].aggregate["ev_pct"]),
        "label": groups.attrs.get("label", "user-supplied groups"),
        "is_synthetic": synthetic,
        "source": groups.attrs.get("source", ""),
        "household_elasticity": float(household_elasticity),
    }
    return out


def compute_sensitivity_grid(
    config: TradeBriefConfig,
    icio: ICIOData,
    calib: Any,
    groups: pd.DataFrame,
    baseline_sol: BriefSolution | None = None,
) -> pd.DataFrame:
    """Compute sensitivity grid across trade elasticities sigma and household elasticities sigma_h."""
    rows: list[dict[str, Any]] = []
    trade_elasticities = config.sensitivity_trade_elasticities
    household_elasticities = config.sensitivity_household_elasticities

    sol_cache: dict[float, BriefSolution] = {}
    if baseline_sol is not None:
        sol_cache[float(baseline_sol.settings["sigma"])] = baseline_sol

    for sigma in trade_elasticities:
        s_val = float(sigma)
        if s_val not in sol_cache:
            sol_cache[s_val] = solve_brief(
                config.importer, config.exporter, config.sector, config.tariff_rate,
                sigma=s_val, bundled=(icio, calib)
            )
        sol = sol_cache[s_val]
        nat = national_record(sol)

        for sigma_h in household_elasticities:
            h_val = float(sigma_h)
            hh = household_incidence(sol, groups, household_elasticity=h_val)
            for grp, r in hh.iterrows():
                rows.append({
                    "trade_elasticity": s_val,
                    "household_elasticity": h_val,
                    "group": str(grp),
                    "population_weight": float(r["population_weight"]),
                    "annual_spending": float(r["annual_spending"]),
                    "ev_pct_proportional": float(r["ev_pct_proportional"]),
                    "ev_pct_equal": float(r["ev_pct_equal"]),
                    "price_ev_pct": float(r["price_ev_pct"]),
                    "income_ev_pct": float(r["income_ev_pct"]),
                    "national_ev_pct": float(nat["ev_pct_consumption"]),
                    "national_ev_musd": float(nat["ev_musd"]),
                    "national_terms_of_trade_pct": float(nat["terms_of_trade_pct_importer"]),
                })
            rows.append({
                "trade_elasticity": s_val,
                "household_elasticity": h_val,
                "group": "all",
                "population_weight": 1.0,
                "annual_spending": float(np.average(hh["annual_spending"], weights=hh["population_weight"])),
                "ev_pct_proportional": float(hh.attrs["aggregate_ev_pct_proportional"]),
                "ev_pct_equal": float(hh.attrs["aggregate_ev_pct_equal"]),
                "price_ev_pct": float(np.average(hh["price_ev_pct"], weights=hh["population_weight"])),
                "income_ev_pct": float(np.average(hh["income_ev_pct"], weights=hh["population_weight"])),
                "national_ev_pct": float(nat["ev_pct_consumption"]),
                "national_ev_musd": float(nat["ev_musd"]),
                "national_terms_of_trade_pct": float(nat["terms_of_trade_pct_importer"]),
            })

    return pd.DataFrame(rows)


def compute_brief(
    config: TradeBriefConfig | None = None,
    *,
    data_path: str | Path | None = None,
    importer: str | None = None,
    exporter: str | None = None,
    sector: str | None = None,
    tariff_rate: float | None = None,
    use_enigh: bool | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Solve the trade engine and household incidence, returning (record, paths)."""
    if config is None:
        kwargs: dict[str, Any] = {}
        if data_path is not None:
            kwargs["data_path"] = data_path
        if importer is not None:
            kwargs["importer"] = importer
        if exporter is not None:
            kwargs["exporter"] = exporter
        if sector is not None:
            kwargs["sector"] = sector
        if tariff_rate is not None:
            kwargs["tariff_rate"] = tariff_rate
        if use_enigh is not None:
            kwargs["use_enigh"] = use_enigh
        config = TradeBriefConfig(**kwargs)

    icio, calib = load_bundled()
    validate_scenario(icio, config.importer, config.exporter, config.sector, config.tariff_rate)

    if config.data_path is not None:
        groups = load_groups(config.data_path, icio, config.importer)
    elif config.use_enigh:
        groups = load_enigh_groups(icio, config.importer)
    else:
        groups = synthetic_groups(icio, config.importer)

    sol = solve_brief(
        config.importer, config.exporter, config.sector, config.tariff_rate,
        sigma=config.trade_elasticity, bundled=(icio, calib)
    )
    households = household_incidence(sol, groups, household_elasticity=config.household_elasticity)
    paths = compute_sensitivity_grid(config, icio, calib, groups, baseline_sol=sol)
    record = headline_record(sol, households)
    record["specification"] = asdict(config)
    return record, paths


# --- record, outputs and CLI ------------------------------------------------

DEFAULT_OUTPUT = Path("output") / "first_brief_tariffs"
EXPERIMENT_DESCRIPTION = "substitution across all country-sector intermediate inputs (0 = Leontief)"

_TEXT = {
    "en": {
        "component": "Component", "musd": "million USD", "pct": "% of baseline consumption",
        "rows": ("Purchaser prices", "Factor income", "Fiscal transfers", "Equivalent variation"),
        "headline_note": ("Positive values are gains. The three lines are an endpoint attribution valued "
                          "in the model's numeraire; read purchaser prices and factor income together. "
                          "No retaliation: {exporter} does not respond."),
        "groups_caption": "Household groups: {label}.",
        "group": "Group", "prop": "EV, proportional rebate (% of spending)",
        "equal": "EV, equal rebate (% of spending)", "all": "All groups (spending-weighted)",
        "groups_note": ("Proportional: the change in transfers (tariff revenue) is rebated in proportion "
                        "to baseline spending; equal: the same amount per household."),
        "groups_note_synthetic": (" These groups' spending-weighted basket equals the national basket by "
                                  "construction, so the proportional aggregate matches the national EV."),
        "groups_note_user": (" The aggregate equals the national EV only when the groups' spending-weighted "
                             "basket equals the national basket; compare it with the national row."),
        "national_row": "National EV (% of baseline consumption)",
        "user_label": "user-supplied groups",
        "no_experiment": "No experiment was run (use --experiment sigma=2).",
        "exp_rows": ("Purchaser prices (million USD)", "Factor income (million USD)",
                     "Fiscal transfers (million USD)", "Equivalent variation (million USD)",
                     "Equivalent variation (% of baseline consumption)"),
        "measure": "Measure", "headline_col": "sigma = 0 (headline)", "exp_col": "sigma = {value:g}",
        "exp_note": ("sigma is the substitution elasticity across all country-sector intermediate inputs "
                     "(0 = Leontief, fixed sourcing); final-use origin baskets stay fixed."),
        "title": "First policy brief: {tariff:g}% tariff by {importer} on {what} from {exporter}",
        "all_goods": "all modeled sectors (including services)", "sector_goods": "sector {sector} output",
        "table": "Table: {citation}; {unit}, {year}.",
        "panel_national": "National EV split ({importer})", "panel_groups": "Group EV",
        "axis_pct_cons": "% of baseline consumption", "axis_pct_spend": "% of group spending",
        "legend_prop": "proportional rebate", "legend_equal": "equal rebate",
        "written": "Outputs written to {path}",
    },
    "es": {
        "component": "Componente", "musd": "millones de USD", "pct": "% del consumo base",
        "rows": ("Precios de comprador", "Ingreso de los factores", "Transferencias fiscales",
                 "Variación equivalente"),
        "headline_note": ("Los valores positivos son ganancias. Las tres líneas son una atribución entre "
                          "los dos equilibrios valorada en el numerario del modelo; lea juntos los precios "
                          "de comprador y el ingreso de los factores. Sin represalias: {exporter} no responde."),
        "groups_caption": "Grupos de hogares: {label}.",
        "group": "Grupo", "prop": "VE, reembolso proporcional (% del gasto)",
        "equal": "VE, reembolso igualitario (% del gasto)", "all": "Todos los grupos (ponderado por gasto)",
        "groups_note": ("Proporcional: el cambio en las transferencias (ingreso arancelario) se reembolsa en "
                        "proporción al gasto base; igualitario: el mismo monto por hogar."),
        "groups_note_synthetic": (" La canasta media de estos grupos, ponderada por gasto, es igual a la "
                                  "nacional por construcción, por lo que el agregado proporcional coincide con "
                                  "la VE nacional."),
        "groups_note_user": (" El agregado es igual a la VE nacional solo cuando la canasta media de los "
                             "grupos, ponderada por gasto, es igual a la nacional; compárelo con la fila "
                             "nacional."),
        "national_row": "VE nacional (% del consumo base)",
        "user_label": "grupos proporcionados por el usuario",
        "no_experiment": "No se ejecutó ningún experimento (use --experiment sigma=2).",
        "exp_rows": ("Precios de comprador (millones de USD)", "Ingreso de los factores (millones de USD)",
                     "Transferencias fiscales (millones de USD)", "Variación equivalente (millones de USD)",
                     "Variación equivalente (% del consumo base)"),
        "measure": "Medida", "headline_col": "sigma = 0 (principal)", "exp_col": "sigma = {value:g}",
        "exp_note": ("sigma es la elasticidad de sustitución entre todos los insumos intermedios país-sector "
                     "(0 = Leontief, abastecimiento fijo); las canastas de origen del uso final no cambian."),
        "title": "Primer informe de política: arancel de {tariff:g}% de {importer} a {what} de {exporter}",
        "all_goods": "todos los sectores modelados (incluidos servicios)", "sector_goods": "la producción del sector {sector}",
        "table": "Tabla: {citation}; {unit}, {year}.",
        "panel_national": "Descomposición de la VE nacional ({importer})", "panel_groups": "VE por grupo",
        "axis_pct_cons": "% del consumo base", "axis_pct_spend": "% del gasto del grupo",
        "legend_prop": "reembolso proporcional", "legend_equal": "reembolso igualitario",
        "written": "Resultados escritos en {path}",
    },
}


def _lang(lang: str) -> dict[str, Any]:
    if lang not in _TEXT:
        raise ValueError(f"lang {lang!r} must be 'en' or 'es'")
    return _TEXT[lang]


def _fmt_musd(value: float, lang: str) -> str:
    text = f"{value:,.0f}"
    return text.replace(",", " ") if lang == "es" else text


def _fmt_pct(value: float, lang: str) -> str:
    text = f"{value:.4f}"
    return text.replace(".", ",") if lang == "es" else text


def _label(households: dict[str, Any], lang: str) -> str:
    if households.get("is_synthetic"):
        return SYNTHETIC_LABEL_ES if lang == "es" else SYNTHETIC_LABEL
    lbl = households.get("label")
    if lbl and lbl != "user-supplied groups":
        if "ENIGH 2022" in str(lbl):
            return "Quintiles de ingreso ENIGH 2022 (microdatos INEGI)" if lang == "es" else str(lbl)
        return str(lbl)
    return _lang(lang)["user_label"]


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.name
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def headline_record(sol: BriefSolution, households: pd.DataFrame,
                    experiment: BriefSolution | None = None) -> dict[str, Any]:
    """The reproducible record behind every table: no timings, no versions."""
    meta = sol.icio.metadata
    record = {
        "schema": 1,
        "scenario": {k: sol.settings[k] for k in ("importer", "exporter", "sector", "tariff")},
        "solver": {**SOLVER, "sigma": float(sol.settings["sigma"])},
        "table": {"citation": meta["citation"], "unit": meta["unit"], "year": meta["year"]},
        "national": national_record(sol),
        "households": {
            "label": households.attrs["label"],
            "is_synthetic": bool(households.attrs["is_synthetic"]),
            "rows": [{"group": str(g), "ev_pct_proportional": float(row["ev_pct_proportional"]),
                      "ev_pct_equal": float(row["ev_pct_equal"])} for g, row in households.iterrows()],
            "aggregate_ev_pct_proportional": float(households.attrs["aggregate_ev_pct_proportional"]),
            "aggregate_ev_pct_equal": float(households.attrs["aggregate_ev_pct_equal"]),
        },
        "experiment": None,
    }
    if experiment is not None:
        record["experiment"] = {"name": "sigma", "value": float(experiment.settings["sigma"]),
                                "description": EXPERIMENT_DESCRIPTION,
                                "national": national_record(experiment)}
    return _jsonable(record)


def _national_frame(national: dict[str, float], lang: str) -> pd.DataFrame:
    t = _lang(lang)
    keys = ("price_effect_musd", "factor_income_effect_musd", "fiscal_transfer_effect_musd", "ev_musd")
    base = national["consumption_base_musd"]
    frame = pd.DataFrame({t["musd"]: [national[k] for k in keys],
                          t["pct"]: [100.0 * national[k] / base for k in keys]},
                         index=pd.Index(t["rows"], name=t["component"]))
    return frame


def _households_frame(households: dict[str, Any], lang: str) -> pd.DataFrame:
    t = _lang(lang)
    rows = [(r["group"], r["ev_pct_proportional"], r["ev_pct_equal"]) for r in households["rows"]]
    rows.append((t["all"], households["aggregate_ev_pct_proportional"],
                 households["aggregate_ev_pct_equal"]))
    frame = pd.DataFrame(rows, columns=[t["group"], t["prop"], t["equal"]]).set_index(t["group"])
    return frame


def _experiment_frame(record: dict[str, Any], lang: str) -> pd.DataFrame:
    t = _lang(lang)
    exp = record["experiment"]
    keys = ("price_effect_musd", "factor_income_effect_musd", "fiscal_transfer_effect_musd",
            "ev_musd", "ev_pct_consumption")
    return pd.DataFrame({t["headline_col"]: [record["national"][k] for k in keys],
                         t["exp_col"].format(value=exp["value"]): [exp["national"][k] for k in keys]},
                        index=pd.Index(t["exp_rows"], name=t["measure"]))


def _markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join([":---"] + ["---:"] * (len(header) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _num(value: float, digits: int, lang: str, *, sign: bool = False) -> str:
    text = f"{value:+.{digits}f}" if sign else f"{value:.{digits}f}"
    return text.replace(".", ",") if lang == "es" else text


def _require_default_scenario(record: dict[str, Any], block: str) -> None:
    sc = record["scenario"]
    if (sc["importer"], sc["exporter"], sc["sector"]) != ("USA", "CHN", "all"):
        raise ValueError(f"block {block!r} describes the docs page's default scenario "
                         "(USA tariff on all goods from CHN) only")


def _require(condition: bool, block: str) -> None:
    if not condition:
        raise ValueError(f"block {block!r}: the prose no longer matches the record's signs; rewrite it")


def _headline_reading(record: dict[str, Any], lang: str) -> str:
    _require_default_scenario(record, "headline_reading")
    nat = record["national"]
    base = nat["consumption_base_musd"]
    pf = -100.0 * (nat["price_effect_musd"] + nat["factor_income_effect_musd"]) / base
    fiscal = 100.0 * nat["fiscal_transfer_effect_musd"] / base
    ev, relw = nat["ev_pct_consumption"], nat["relative_wage_pct"]
    tot_i, tot_e = nat["terms_of_trade_pct_importer"], nat["terms_of_trade_pct_exporter"]
    _require(nat["price_effect_musd"] < 0 and nat["factor_income_effect_musd"] < 0 and fiscal > 0
             and ev > 0 and relw < 0 and tot_i < 0 and tot_e >= 0, "headline_reading")
    v = {"pf": _num(pf, 3, lang), "fiscal": _num(fiscal, 3, lang), "ev": _num(ev, 3, lang, sign=True),
         "relw": _num(abs(relw), 2, lang), "tot": _num(abs(tot_i), 2, lang),
         "cons": _fmt_musd(nat["ev_musd"], lang),
         "inv": _fmt_musd(nat["investment_change_base_prices_musd"], lang),
         "abroad": _fmt_musd(nat["abroad_change_base_prices_musd"], lang)}
    if lang == "es":
        return ("Cómo leerla: el alza de los precios de comprador y la caída del ingreso de los factores "
                f"de Estados Unidos cuestan juntas {v['pf']}% del consumo base. El ingreso arancelario, "
                "devuelto como suma fija a los residentes de Estados Unidos, agrega "
                f"{v['fiscal']}%, de modo que la VE neta es pequeña y positiva ({v['ev']}%). No es una "
                "ganancia de términos de intercambio. Con abastecimiento fijo, los productores chinos no "
                f"bajan sus precios: los salarios de Estados Unidos caen alrededor de {v['relw']}% respecto "
                "a los salarios chinos, y los términos de intercambio de Estados Unidos (valores unitarios "
                "de exportación sobre los de importación, antes de aranceles) empeoran alrededor de "
                f"{v['tot']}%. La mejora del consumo de {v['cons']} millones de USD acompaña cambios de "
                f"{v['inv']} en inversión y {v['abroad']} en compras de residentes en el exterior, "
                "valorados a precios base. La VE valora solo el consumo: la asignación entre usos "
                "finales cambia. Un solucionador independiente reproduce el resultado y comprueba "
                "el balance mundial de recursos; vea [la validación](tariff_validation.md). "
                "La descomposición de la VE es una atribución entre equilibrios, no canales causales "
                "separados. El resultado es además unilateral: China no toma represalias "
                "aquí, y una respuesta lo cambiaría.")
    return ("Read it this way: higher purchaser prices and lower US factor income together cost "
            f"{v['pf']}% of baseline consumption. The tariff revenue, rebated lump sum to US residents, "
            f"adds {v['fiscal']}%, so the net EV is small and positive ({v['ev']}%). This is not a "
            "terms-of-trade gain. With fixed sourcing, Chinese producers do not cut their prices: US "
            f"wages fall by about {v['relw']}% relative to Chinese wages, and the US terms of trade "
            "(export over import unit values, before duties) worsen by about "
            f"{v['tot']}%. The consumption improvement of {v['cons']} million USD accompanies changes "
            f"of {v['inv']} in investment and {v['abroad']} in residents' purchases abroad, valued at "
            "baseline prices. EV values consumption only: allocation across final uses changes. "
            "An independent solver reproduces the result and checks the world resource balance; "
            "see [the validation](tariff_validation.md). The EV split is an endpoint attribution, "
            "not separate causal channels. The result is also unilateral: China does not retaliate here, and a "
            "response would change it.")


def _experiment_reading(record: dict[str, Any], lang: str) -> str:
    _require_default_scenario(record, "experiment_reading")
    exp = record["experiment"]
    if exp is None:
        raise ValueError("block 'experiment_reading' needs a record with an experiment")
    n0, n2 = record["national"], exp["national"]
    pf0 = n0["price_effect_musd"] + n0["factor_income_effect_musd"]
    pf2 = n2["price_effect_musd"] + n2["factor_income_effect_musd"]
    _require(n2["ev_pct_consumption"] > n0["ev_pct_consumption"] > 0
             and n2["wage_pct_exporter"] < 0 < n2["wage_pct_importer"]
             and n2["terms_of_trade_pct_importer"] > 0
             and n2["price_effect_musd"] < 0 < n2["factor_income_effect_musd"]
             and pf0 < pf2 < 0
             and n2["tariff_rebate_effect_musd"] < n0["tariff_rebate_effect_musd"]
             and n2["fiscal_transfer_effect_musd"] > n0["fiscal_transfer_effect_musd"], "experiment_reading")
    def m(x: float) -> str:
        return _fmt_musd(x, lang)

    v = {"value": _num(exp["value"], 0, lang),
         "ev2": _num(n2["ev_pct_consumption"], 3, lang, sign=True),
         "ev0": _num(n0["ev_pct_consumption"], 3, lang, sign=True),
         "we": _num(abs(n2["wage_pct_exporter"]), 1, lang), "wi": _num(n2["wage_pct_importer"], 1, lang),
         "relw": _num(n2["relative_wage_pct"], 1, lang), "tot": _num(n2["terms_of_trade_pct_importer"], 1, lang),
         "p2": m(n2["price_effect_musd"]), "f2": m(n2["factor_income_effect_musd"]),
         "pf2": m(pf2), "pf0": m(pf0), "tr0": m(n0["tariff_rebate_effect_musd"]),
         "tr2": m(n2["tariff_rebate_effect_musd"]), "fi0": m(n0["fiscal_transfer_effect_musd"]),
         "fi2": m(n2["fiscal_transfer_effect_musd"])}
    if lang == "es":
        return (f"Con `sigma={v['value']}` la ganancia de Estados Unidos es varias veces mayor ({v['ev2']}% "
                f"frente a {v['ev0']}% del consumo base). `sigma` se aplica a los compradores de insumos "
                "intermedios de todos los países, no solo de Estados Unidos. Los salarios chinos caen "
                f"alrededor de {v['we']}% y los de Estados Unidos suben alrededor de {v['wi']}% (ambos en el "
                f"numerario del modelo; los de Estados Unidos suben {v['relw']}% respecto a los chinos), y los términos de intercambio de "
                f"Estados Unidos mejoran alrededor de {v['tot']}%: aquí sí aparece un efecto de términos de "
                "intercambio, a diferencia del resultado principal. Las líneas de precios de comprador y de "
                f"ingreso de los factores son grandes y se compensan ({v['p2']} y {v['f2']} millones de USD), "
                f"así que lea su suma ({v['pf2']} frente a {v['pf0']}). La parte de aranceles a la importación "
                f"de la línea fiscal baja ligeramente (de {v['tr0']} a {v['tr2']} millones de USD), mientras "
                f"la línea fiscal sube (de {v['fi0']} a {v['fi2']}) porque crecen otros ingresos tributarios "
                "devueltos. `sigma` no es una elasticidad comercial (las canastas de origen del uso final "
                "no cambian) y la tabla no la identifica: varíela como un supuesto, no la trate como una "
                "constante conocida.")
    return (f"With `sigma={v['value']}` the US gain is several times larger ({v['ev2']}% against "
            f"{v['ev0']}% of baseline consumption). `sigma` applies to intermediate buyers in every country, "
            f"not only in the US. Chinese wages fall by about {v['we']}% and US wages rise by about "
            f"{v['wi']}% (both in the model's numeraire; US wages rise {v['relw']}% relative to Chinese "
            "wages), and the US "
            f"terms of trade improve by about {v['tot']}%: a terms-of-trade effect appears here, not in the "
            "headline. The purchaser-price and factor-income lines are large and offsetting "
            f"({v['p2']} and {v['f2']} million USD), so read their sum ({v['pf2']} against {v['pf0']}). "
            f"The import-duty part of the fiscal line falls slightly ({v['tr0']} to {v['tr2']} million USD), "
            f"while the fiscal line rises ({v['fi0']} to {v['fi2']}) because other rebated tax receipts "
            "grow. `sigma` is not a trade elasticity (final-use origin baskets stay fixed), and the table "
            "does not identify it: vary it as an assumption, not as a known constant.")


def page_block(record: dict[str, Any],
               block: Literal["headline", "headline_reading", "households", "experiment",
                              "experiment_reading"],
               lang: str) -> str:
    """A markdown table plus a one-sentence note, as the docs pages show it."""
    t = _lang(lang)
    if block == "headline":
        frame = _national_frame(record["national"], lang)
        rows: list[tuple[str, ...]] = [
            (str(name), _fmt_musd(row.iloc[0], lang), _fmt_pct(row.iloc[1], lang))
            for name, row in frame.iterrows()]
        rows[-1] = tuple(f"**{cell}**" for cell in rows[-1])
        note = t["headline_note"].format(exporter=record["scenario"]["exporter"])
        return _markdown_table([t["component"], t["musd"], t["pct"]], rows) + "\n\n" + note
    if block == "headline_reading":
        return _headline_reading(record, lang)
    if block == "experiment_reading":
        return _experiment_reading(record, lang)
    if block == "households":
        hh = record["households"]
        frame = _households_frame(hh, lang)
        rows = [(name, _fmt_pct(row.iloc[0], lang), _fmt_pct(row.iloc[1], lang))
                for name, row in frame.iterrows()]
        note = t["groups_note"]
        if hh["is_synthetic"]:
            note += t["groups_note_synthetic"]
        else:
            national = _fmt_pct(record["national"]["ev_pct_consumption"], lang)
            rows.append((t["national_row"], national, national))
            note += t["groups_note_user"]
        caption = t["groups_caption"].format(label=_label(hh, lang))
        return (caption + "\n\n" + _markdown_table([t["group"], t["prop"], t["equal"]], rows)
                + "\n\n" + note)
    if block == "experiment":
        if record["experiment"] is None:
            return t["no_experiment"]
        frame = _experiment_frame(record, lang)
        fmt = [_fmt_musd] * 4 + [_fmt_pct]
        rows = [(name, f(row.iloc[0], lang), f(row.iloc[1], lang))
                for f, (name, row) in zip(fmt, frame.iterrows())]
        return _markdown_table([t["measure"], *frame.columns], rows) + "\n\n" + t["exp_note"]
    raise ValueError(f"block {block!r} must be 'headline', 'headline_reading', 'households', "
                     "'experiment' or 'experiment_reading'")


def _title(record: dict[str, Any], lang: str) -> str:
    t = _lang(lang)
    sc = record["scenario"]
    what = t["all_goods"] if sc["sector"] == "all" else t["sector_goods"].format(sector=sc["sector"])
    return t["title"].format(tariff=100 * sc["tariff"], importer=sc["importer"], what=what,
                             exporter=sc["exporter"])


def summary_text(record: dict[str, Any], lang: str = "en") -> str:
    """The plain-text summary ``main`` prints."""
    t = _lang(lang)
    table = record["table"]
    parts = [_title(record, lang),
             t["table"].format(citation=table["citation"], unit=table["unit"], year=table["year"]),
             page_block(record, "headline", lang),
             page_block(record, "households", lang)]
    if record["experiment"] is not None:
        parts.append(page_block(record, "experiment", lang))
    return "\n\n".join(parts) + "\n"


def _figure(record: dict[str, Any], lang: str, path: Path) -> None:
    from matplotlib.figure import Figure

    t = _lang(lang)
    fig = Figure(figsize=(10, 4.2), dpi=100)
    left, right = fig.subplots(1, 2, gridspec_kw={"width_ratios": [1.1, 1.0]})
    national = _national_frame(record["national"], lang).iloc[:, 1]
    colors = ["#4C72B0", "#DD8452", "#55A868", "0.2"]
    left.barh(list(national.index)[::-1], national.to_numpy()[::-1], color=colors[::-1])
    left.axvline(0, color="0.1", linewidth=0.7)
    left.set_xlabel(t["axis_pct_cons"])
    left.set_title(t["panel_national"].format(importer=record["scenario"]["importer"]), fontsize=10)
    hh = record["households"]
    groups = [r["group"] for r in hh["rows"]]
    y = np.arange(len(groups))
    right.barh(y - 0.2, [r["ev_pct_proportional"] for r in hh["rows"]], height=0.38,
               color="#4C72B0", label=t["legend_prop"])
    right.barh(y + 0.2, [r["ev_pct_equal"] for r in hh["rows"]], height=0.38,
               color="#C44E52", label=t["legend_equal"])
    right.set_yticks(y, groups)
    right.invert_yaxis()
    right.axvline(0, color="0.1", linewidth=0.7)
    right.set_xlabel(t["axis_pct_spend"])
    right.set_title(f"{t['panel_groups']}\n" + textwrap.fill(_label(hh, lang), 48), fontsize=9)
    right.legend(fontsize=8, loc="best")
    if lang == "es":
        from matplotlib.ticker import FuncFormatter
        for axis in (left, right):
            axis.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:g}".replace(".", ",")))
    fig.suptitle(textwrap.fill(_title(record, lang), 95), fontsize=11)
    fig.tight_layout()
    fig.savefig(path, metadata={"Software": None})


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_average() -> list[float] | None:
    getloadavg = getattr(os, "getloadavg", None)  # not available on Windows
    if getloadavg is None:
        return None
    try:
        return [round(x, 2) for x in getloadavg()]
    except OSError:
        return None


def _convergence(state: Any) -> dict[str, Any]:
    meta = state.metadata
    return {"converged": bool(state.converged), "solver_termination": meta.get("solver_termination"),
            "account_residuals": meta.get("account_residuals")}


def _build_headline_markdown(
    record: dict[str, Any],
    households: pd.DataFrame,
    paths: pd.DataFrame | None,
    lang: str = "en",
) -> str:
    t = _lang(lang)
    parts = []
    parts.append(f"# {_title(record, lang)}")
    parts.append(t["table"].format(**record["table"]))
    parts.append(page_block(record, "headline", lang))
    sc = record["scenario"]
    if (sc["importer"], sc["exporter"], sc["sector"]) == ("USA", "CHN", "all") and record["solver"].get("sigma", 0.0) == 0.0:
        try:
            parts.append(_headline_reading(record, lang))
        except ValueError:
            pass
    hh_title = "## Incidencia en los hogares" if lang == "es" else "## Household incidence"
    parts.append(hh_title)
    parts.append(page_block(record, "households", lang))
    if paths is not None:
        if lang == "es":
            sens_heading = "## Sensibilidad a elasticidades comerciales y de hogares"
            sens_note = ("sigma es la elasticidad de sustitución de insumos intermedios en el comercio "
                         "(0 = Leontief, abastecimiento fijo); sigma_h es la elasticidad de sustitución "
                         "en las preferencias homotéticas CES de los hogares.")
            t_trade = "Elasticidad comercial (sigma)"
            t_hh = "Elasticidad hogares (sigma_h)"
            t_nat = "VE nacional (% consumo)"
            t_q1 = "VE Grupo inferior (% gasto)"
            t_q5 = "VE Grupo superior (% gasto)"
        else:
            sens_heading = "## Sensitivity to trade and household elasticities"
            sens_note = ("sigma is the substitution elasticity across intermediate inputs in trade "
                         "(0 = Leontief, fixed sourcing); sigma_h is the substitution elasticity "
                         "in homothetic CES household preferences.")
            t_trade = "Trade elasticity (sigma)"
            t_hh = "Household elasticity (sigma_h)"
            t_nat = "National EV (% cons)"
            t_q1 = "Bottom group EV (% spend)"
            t_q5 = "Top group EV (% spend)"

        headers = [t_trade, t_hh, t_nat, t_q1, t_q5]
        s_rows = []
        pairs = sorted(set(zip(paths["trade_elasticity"], paths["household_elasticity"])))
        for s_val, h_val in pairs:
            sub = paths[(paths["trade_elasticity"] == s_val) & (paths["household_elasticity"] == h_val)]
            nat_ev = float(sub["national_ev_pct"].iloc[0])
            group_rows = sub[sub["group"] != "all"]
            q1_ev = float(group_rows["ev_pct_proportional"].iloc[0]) if len(group_rows) > 0 else nat_ev
            q5_ev = float(group_rows["ev_pct_proportional"].iloc[-1]) if len(group_rows) > 0 else nat_ev
            s_rows.append([
                _num(s_val, 1, lang),
                _num(h_val, 1, lang),
                _fmt_pct(nat_ev, lang),
                _fmt_pct(q1_ev, lang),
                _fmt_pct(q5_ev, lang),
            ])
        parts.append(sens_heading)
        parts.append(_markdown_table(headers, s_rows))
        parts.append(sens_note)

    return "\n\n".join(parts) + "\n"


def _build_headline_csv(
    record: dict[str, Any],
    households: pd.DataFrame,
    paths: pd.DataFrame | None,
) -> str:
    rows = []
    nat = record["national"]
    for k in ("price_effect_musd", "factor_income_effect_musd", "fiscal_transfer_effect_musd",
              "ev_musd", "ev_pct_consumption", "terms_of_trade_pct_importer", "relative_wage_pct"):
        rows.append({"domain": "national", "item": k, "value": nat.get(k)})
    hh = record["households"]
    for r in hh["rows"]:
        rows.append({"domain": "household", "item": f"{r['group']}_ev_pct_proportional", "value": r["ev_pct_proportional"]})
        rows.append({"domain": "household", "item": f"{r['group']}_ev_pct_equal", "value": r["ev_pct_equal"]})
    rows.append({"domain": "household", "item": "aggregate_ev_pct_proportional", "value": hh["aggregate_ev_pct_proportional"]})
    rows.append({"domain": "household", "item": "aggregate_ev_pct_equal", "value": hh["aggregate_ev_pct_equal"]})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False)


def _build_headline_tex(
    record: dict[str, Any],
    households: pd.DataFrame,
    paths: pd.DataFrame | None,
    lang: str = "en",
) -> str:
    t = _lang(lang)
    nat_frame = _national_frame(record["national"], lang)
    hh_frame = _households_frame(record["households"], lang)
    tex_nat = f"% {t['table'].format(**record['table'])}\n" + df_to_latex(nat_frame, digits=4)
    tex_hh = f"% {t['groups_caption'].format(label=_label(record['households'], lang))}\n" + df_to_latex(hh_frame, digits=4)
    return tex_nat + "\n\n" + tex_hh + "\n"


def write_outputs(sol: BriefSolution, households: pd.DataFrame, output: str | Path, *,
                  lang: str = "en", experiment: BriefSolution | None = None,
                  paths: pd.DataFrame | None = None, config: TradeBriefConfig | None = None,
                  headline_files: bool = False) -> dict[str, Any]:
    """Write tables (md/csv/tex), the figure, ``headline.json`` and ``manifest.json``."""
    t = _lang(lang)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = headline_record(sol, households, experiment)
    if config is not None:
        record["config"] = _jsonable(asdict(config))
    hh = record["households"]
    label = _label(hh, lang)
    citation_line = t["table"].format(**record["table"])
    files: list[str] = []

    def write(name: str, text: str) -> None:
        (output / name).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        files.append(name)

    national = _national_frame(record["national"], lang)
    write("national.md", f"# {_title(record, lang)}\n\n{page_block(record, 'headline', lang)}\n\n{citation_line}")
    write("national.csv", national.to_csv())
    write("national.tex", f"% {citation_line}\n" + df_to_latex(national, digits=4))

    detail = households.copy()
    detail.insert(0, "label", label)
    write("households.md", page_block(record, "households", lang))
    write("households.csv", detail.to_csv())
    write("households.tex", f"% {t['groups_caption'].format(label=label)}\n"
                            + df_to_latex(_households_frame(hh, lang), digits=4))
    if experiment is not None:
        exp = _experiment_frame(record, lang)
        write("experiment.md", page_block(record, "experiment", lang))
        write("experiment.csv", exp.to_csv())
        write("experiment.tex", f"% {t['exp_note']}\n" + df_to_latex(exp, digits=4))
    else:
        # These are generated artifacts owned by this example. A new baseline
        # run must not leave an old experiment beside its current manifest.
        for suffix in ("md", "csv", "tex"):
            (output / f"experiment.{suffix}").unlink(missing_ok=True)

    if paths is not None or config is not None or headline_files:
        if paths is not None:
            write("paths.csv", paths.to_csv(index=False))
        write("headline.md", _build_headline_markdown(record, households, paths, lang))
        write("headline.csv", _build_headline_csv(record, households, paths))
        write("headline.tex", _build_headline_tex(record, households, paths, lang))
        if config is not None and config.bilingual:
            alt_lang = "es" if lang == "en" else "en"
            (output / f"headline_{alt_lang}.md").write_text(
                _build_headline_markdown(record, households, paths, alt_lang) + "\n", encoding="utf-8"
            )
            files.append(f"headline_{alt_lang}.md")

    _figure(record, lang, output / "first_brief.png")
    files.append("first_brief.png")
    write("headline.json", json.dumps(record, indent=2, ensure_ascii=False))

    meta = sol.icio.metadata
    table_path = Path(meta["file_path"])
    timings = {**sol.timings, "households": float(households.attrs.get("elapsed_seconds", 0.0))}
    convergence = {"baseline": _convergence(sol.base), "counterfactual": _convergence(sol.counterfactual)}
    if experiment is not None:
        timings["experiment"] = float(sum(experiment.timings.values()))
        convergence["experiment_baseline"] = _convergence(experiment.base)
        convergence["experiment_counterfactual"] = _convergence(experiment.counterfactual)

    groups_info: dict[str, Any] = {
        "source": households.attrs.get("source", ""),
        "label": households.attrs.get("label", ""),
        "is_synthetic": bool(households.attrs.get("is_synthetic", False)),
    }
    if households.attrs.get("file"):
        groups_info["file"] = Path(households.attrs["file"]).name
    elif households.attrs.get("source"):
        src = str(households.attrs["source"])
        if src.endswith(".npz") or src.endswith(".csv"):
            groups_info["file"] = Path(src).name
    if not households.attrs.get("is_synthetic", False) and "file" not in groups_info:
        groups_info["file"] = "enigh2022_household_policy.npz"
    if households.attrs.get("sha256"):
        groups_info["sha256"] = households.attrs["sha256"]

    manifest_dict: dict[str, Any] = {
        "package_version": puremacro.__version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "table": {"file": table_path.name, "citation": meta["citation"], "unit": meta["unit"],
                  "year": meta["year"], "source_md5": meta.get("source_md5"),
                  "sha256": _sha256(table_path)},
        "scenario": record["scenario"],
        "solver": record["solver"],
        "convergence": convergence,
        "timings": timings,
        "experiment": None if experiment is None else {
            "name": "sigma", "value": float(experiment.settings["sigma"]),
            "description": EXPERIMENT_DESCRIPTION, "solver": {**SOLVER, "sigma": float(experiment.settings["sigma"])}},
        "groups": groups_info,
        "lang": lang,
        "load_average": _load_average(),
        "environment": environment(),
        "artifacts": {name: {"sha256": _sha256(output / name), "bytes": (output / name).stat().st_size}
                      for name in files},
    }
    if config is not None:
        manifest_dict["config"] = _jsonable(asdict(config))
    manifest = _jsonable(manifest_dict)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8")
    return manifest


def run_brief(output: str | Path = DEFAULT_OUTPUT, *, importer: str = "USA", exporter: str = "CHN",
              sector: str = "all", tariff: float = 0.10, experiment: str | None = None,
              groups: str | Path | None = None, lang: str = "en",
              config: TradeBriefConfig | None = None, use_enigh: bool = False) -> dict[str, Any]:
    """Solve everything, then write the outputs; nothing is written unless every state converged."""
    _lang(lang)
    if config is not None or use_enigh:
        if config is None:
            config = TradeBriefConfig(
                importer=importer, exporter=exporter, sector=sector, tariff_rate=tariff,
                use_enigh=True, data_path=groups,
            )
        out_dir = Path(output) if output is not None else Path(config.output_dir or DEFAULT_OUTPUT)
        icio, calib = load_bundled()
        validate_scenario(icio, config.importer, config.exporter, config.sector, config.tariff_rate)
        if config.data_path is not None:
            frame = load_groups(config.data_path, icio, config.importer)
        elif config.use_enigh:
            frame = load_enigh_groups(icio, config.importer)
        else:
            frame = synthetic_groups(icio, config.importer)
        sol = solve_brief(
            config.importer, config.exporter, config.sector, config.tariff_rate,
            sigma=config.trade_elasticity, bundled=(icio, calib),
        )
        households = household_incidence(sol, frame, household_elasticity=config.household_elasticity)
        paths = compute_sensitivity_grid(config, icio, calib, frame, baseline_sol=sol)
        manifest = write_outputs(sol, households, out_dir, lang=lang, paths=paths, config=config)
        rec = headline_record(sol, households)
        rec["specification"] = asdict(config)
        return {
            "record": rec,
            "paths": paths,
            "solution": sol,
            "households": households,
            "manifest": manifest,
        }

    # Legacy default execution
    sigma = None if experiment is None else parse_experiment(experiment)
    t0 = time.perf_counter()
    icio = _load_table()
    t_load = time.perf_counter() - t0
    t0 = time.perf_counter()
    calib = calibrate_trade_model(icio.matrix, country_codes=icio.country_codes,
                                  sector_codes=icio.sector_codes)
    t_calib = time.perf_counter() - t0
    bundled = (icio, calib)
    validate_scenario(icio, importer, exporter, sector, tariff)
    frame = (synthetic_groups(icio, importer) if groups is None else
             read_groups_csv(groups, icio.sector_codes, supported_sectors(icio, importer)))
    sol = solve_brief(importer, exporter, sector, tariff, bundled=bundled)
    sol.timings.update(load=t_load, calibration=t_calib)
    households = household_incidence(sol, frame)
    alt = (None if sigma is None else
           solve_brief(importer, exporter, sector, tariff, sigma=sigma, bundled=bundled))
    manifest = write_outputs(sol, households, output, lang=lang, experiment=alt)
    return {"record": headline_record(sol, households, alt), "solution": sol,
            "households": households, "experiment": alt, "manifest": manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m puremacro.examples.first_brief_tariffs",
        description="Your first policy brief: a tariff experiment on the bundled OECD ICIO 2019 table.")
    parser.add_argument("--importer", default="USA", help="country levying the tariff (default USA)")
    parser.add_argument("--exporter", default="CHN", help="country whose goods are tariffed (default CHN)")
    parser.add_argument("--sector", default="all", help="'all' or one ISIC section code A..T (default all)")
    parser.add_argument("--tariff", type=float, default=0.10, help="ad valorem rate, 0.10 = 10%% (default)")
    parser.add_argument("--experiment", default="none",
                        help="'none' (default) or 'sigma=<value>', e.g. sigma=2")
    parser.add_argument("--groups", type=Path, default=None,
                        help="CSV of household groups (group, population_weight, annual_spending, A..T)")
    parser.add_argument("--enigh", action="store_true",
                        help="use real ENIGH 2022 household income quintiles (bundled microdata)")
    parser.add_argument("--trade-elasticity", type=float, default=0.0,
                        help="intermediate input substitution elasticity sigma (default 0.0)")
    parser.add_argument("--household-elasticity", type=float, default=0.0,
                        help="household preference elasticity sigma_h (default 0.0, fixed baskets)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"output directory (default {DEFAULT_OUTPUT})")
    parser.add_argument("--lang", choices=("en", "es"), default="en", help="language of the outputs")
    args = parser.parse_args(argv)
    try:
        sigma = parse_experiment(args.experiment)
        icio = _load_table()
        validate_scenario(icio, args.importer, args.exporter, args.sector, args.tariff)
        if args.groups is not None:
            if not args.groups.is_file():
                raise ValueError(f"groups file {str(args.groups)!r} does not exist")
            read_groups_csv(args.groups, icio.sector_codes, supported_sectors(icio, args.importer))
    except ValueError as exc:
        parser.error(str(exc))
    if args.enigh or args.trade_elasticity > 0 or args.household_elasticity > 0:
        cfg = TradeBriefConfig(
            importer=args.importer, exporter=args.exporter, sector=args.sector,
            tariff_rate=args.tariff, trade_elasticity=args.trade_elasticity or 4.0,
            household_elasticity=args.household_elasticity or 1.0,
            use_enigh=args.enigh or (args.groups is None),
            data_path=args.groups, output_dir=args.output,
        )
        study = run_brief(args.output, config=cfg, lang=args.lang)
    else:
        study = run_brief(args.output, importer=args.importer, exporter=args.exporter, sector=args.sector,
                          tariff=args.tariff, experiment=None if sigma is None else args.experiment,
                          groups=args.groups, lang=args.lang)
    print(summary_text(study["record"], args.lang))
    print(_lang(args.lang)["written"].format(path=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
