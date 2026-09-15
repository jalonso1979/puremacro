"""Adversarial Empirical Stress Tests for Milestone 1.

Challenger 1 Suite targeting:
1. .to_typst() across all 11 trade result classes + VAR/DSGE with special characters
   (#, $, *, _, [, ], <, >, ~, @, `, \\) and extreme numbers (NaN, Inf, -Inf, 1e308, 1e-308).
2. Boundary / minimal dimensions: single-country (nc=1), single-sector (ns=1), 1-quarter, 1-player.
3. VFISolution presentation methods (.to_frame(), .summary(), quintet exports, headless .plot())
   across 1D, 2D, None, and mismatched grid configurations.
4. HJBSolution dictionary mapping protocol, frozen immutability (FrozenInstanceError),
   key error behavior, and headless .plot().
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import re
from typing import Any
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge._gradients import ScoreDiagnosticsResult
from puremacro.trade._results import (
    CapacityBottleneckResult,
    GearyKhamisResult,
    JCurveDynamicResult,
    NashTariffResult,
    OptimalTariffResult,
    RetaliationGameResult,
    RevenueRecyclingResult,
    ScenarioBatchResult,
    TradeCalibrationResult,
    TradeEquilibriumResult,
    WelfarePayoffMatrixResult,
)
from puremacro.var.bvar_sv import BVAR_SVForecast
from puremacro.vfi import HJBSolution, VFISolution, solve_hjb_achdou


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Ensure all figures are closed after each test."""
    yield
    plt.close("all")


# ===========================================================================
# Typst Syntax Oracle Validator
# ===========================================================================

def validate_typst_table(typ_str: str) -> dict[str, Any]:
    r"""Empirical oracle for Typst #table(...) syntax correctness.

    Verifies:
    1. Table starts with #table( and ends with matching closing parenthesis.
    2. 'columns: <int>' header is well-formed with n_cols >= 1.
    3. All cells are enclosed in [ ... ].
    4. Brackets inside cell bodies are escaped (\[ and \]) or balanced.
    5. Typst markup specials outside string literals are escaped:
       - No unescaped #
       - No unescaped $
       - No unescaped _
       - No unescaped < or >
       - No unescaped @
       - No unescaped ~
       - No unescaped `
       - Outer bold [* ... *] in header cells is permitted, but inner * must be escaped.
    6. Cell count is a non-zero multiple of columns (every row has exactly n_cols cells).
    """
    assert isinstance(typ_str, str), f"Expected str, got {type(typ_str)}"
    s = typ_str.strip()
    assert s.startswith("#table("), f"Table does not start with '#table(': {s[:40]}"
    assert s.endswith(")"), f"Table does not end with ')': {s[-40:]}"

    # Extract columns parameter
    col_match = re.search(r"columns:\s*(\d+),", s)
    assert col_match is not None, "Missing 'columns: <int>,' declaration"
    n_cols = int(col_match.group(1))
    assert n_cols >= 1, f"Invalid n_cols: {n_cols}"

    # Extract the cell body (after 'columns: N,\n' and before the final ')')
    body_start = col_match.end()
    body = s[body_start:-1].strip()
    if body.endswith(","):
        body = body[:-1].strip()

    # Tokenize cells: each cell is delimited by outer [ ... ]
    # We parse through the body tracking bracket depth
    cells: list[str] = []
    i = 0
    n = len(body)
    while i < n:
        # skip whitespace and commas
        while i < n and (body[i].isspace() or body[i] == ","):
            i += 1
        if i >= n:
            break
        if body[i] != "[":
            raise AssertionError(f"Expected cell starting with '[', got '{body[i:i+20]}' at index {i}")
        
        # Parse until matching unescaped ']'
        start_cell = i
        depth = 0
        in_cell = False
        cell_content = []
        while i < n:
            ch = body[i]
            if ch == "\\" and i + 1 < n:
                # Escaped character: consume both
                cell_content.append(body[i:i+2] )
                i += 2
                continue
            if ch == "[":
                depth += 1
                in_cell = True
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            cell_content.append(ch)
            i += 1

        assert depth == 0, f"Unclosed cell bracket starting at index {start_cell}: {body[start_cell:start_cell+40]}"
        cell_raw = "".join(cell_content)
        # Strip outer [ and ]
        assert cell_raw.startswith("[")
        cells.append(cell_raw[1:])

    assert len(cells) > 0, "Table contains 0 cells"
    assert len(cells) % n_cols == 0, (
        f"Cell count ({len(cells)}) is not a multiple of column count ({n_cols})"
    )

    # Validate cell content escaping
    for idx, cell in enumerate(cells):
        is_header = idx < n_cols
        c = cell.strip()
        
        # If header, it should be wrapped in * ... * for bold
        if is_header and c.startswith("*") and c.endswith("*"):
            inner = c[1:-1].strip()
        else:
            inner = c

        # Scan inner for unescaped specials: #, $, _, <, >, @, ~, `, inner *
        j = 0
        inner_len = len(inner)
        while j < inner_len:
            ch = inner[j]
            if ch == "\\":
                # Escaped character, skip next
                j += 2
                continue
            if ch in ("#", "$", "_", "<", ">", "@", "~", "`"):
                raise AssertionError(
                    f"Unescaped Typst special '{ch}' in cell {idx} ({'header' if is_header else 'data'}): {cell}"
                )
            if ch == "*" and not is_header:
                raise AssertionError(
                    f"Unescaped asterisk '*' in data cell {idx}: {cell}"
                )
            j += 1

    return {
        "n_cols": n_cols,
        "n_cells": len(cells),
        "n_rows": len(cells) // n_cols,
        "cells": cells,
    }


# ===========================================================================
# Adversarial Helper Builders
# ===========================================================================

ADVERSARIAL_CODES = ("US#1", "CA$2", "MX*3", "DE_4", "FR[5]", "GB]6", "IT<7>", "JP>8", "AU~9", "BR@10")
ADVERSARIAL_SECTORS = ("Sec#*", "Sec$_[]", "Sec<~>@")


def _make_adversarial_equilibrium(
    country_codes: tuple[str, ...] = ADVERSARIAL_CODES[:2],
    sector_codes: tuple[str, ...] = ADVERSARIAL_SECTORS[:2],
    values: float = 1.0,
) -> TradeEquilibriumResult:
    nc = len(country_codes)
    ns = len(sector_codes)
    return TradeEquilibriumResult(
        x_sol=np.full(2 * ns * nc + 2 * nc + nc + nc - 1, values),
        p_sol=np.full((1, ns, nc), values),
        y_sol=np.full((1, ns, nc), values * 50.0),
        r_sol=np.full((1, 1, nc), values),
        w_sol=np.full((1, 1, nc), values),
        T_sol=np.full((1, 1, nc), values * 5.0),
        XN_sol=np.array([values]),
        c_sol=np.full((1, 3, nc), values * 20.0),
        pfd_sol=np.full((1, 3, nc), values),
        terms_of_trade=np.full(nc, values),
        country_codes=country_codes,
        sector_codes=sector_codes,
    )


# ===========================================================================
# 1. Challenge .to_typst() Across All 11 Trade Result Classes with Specials
# ===========================================================================

def test_trade_calibration_to_typst_special_characters():
    """Adversarial test: TradeCalibrationResult with special characters in country and sector codes."""
    codes = ("US#1", "CA$2")
    s_codes = ("Sec#*", "Sec$_[]")
    calib = TradeCalibrationResult(
        a=np.ones((4, 2, 2)),
        afd=np.ones((4, 3, 2)),
        alpha=np.full((1, 2, 2), 0.33),
        beta=np.ones((1, 2, 2)),
        k_endow=np.ones((1, 2)),
        l_endow=np.ones((1, 2)),
        invforT=np.zeros((1, 2)),
        tax=np.zeros((1, 2, 2)),
        ytot=np.ones((1, 2, 2)),
        n_countries=2,
        n_sectors=2,
        country_codes=codes,
        sector_codes=s_codes,
    )
    # Detailed
    typ_det = calib.to_typst(detailed=True)
    info_det = validate_typst_table(typ_det)
    assert info_det["n_rows"] == 3  # 1 header + 2 countries
    assert r"US\#1" in typ_det
    assert r"CA\$2" in typ_det

    # Aggregate
    typ_agg = calib.to_typst(detailed=False)
    info_agg = validate_typst_table(typ_agg)
    assert info_agg["n_rows"] == 10  # 1 header + 9 metrics


def test_trade_equilibrium_to_typst_special_characters():
    """Adversarial test: TradeEquilibriumResult with special characters in codes."""
    codes = ("US#1", "CA$2")
    s_codes = ("Sec#*", "Sec$_[]")
    eq = _make_adversarial_equilibrium(country_codes=codes, sector_codes=s_codes)

    typ_det = eq.to_typst(detailed=True)
    info_det = validate_typst_table(typ_det)
    assert info_det["n_rows"] == 3
    assert r"US\#1" in typ_det
    assert r"CA\$2" in typ_det

    typ_agg = eq.to_typst(detailed=False)
    info_agg = validate_typst_table(typ_agg)
    assert info_agg["n_rows"] >= 8


def test_geary_khamis_to_typst_special_characters():
    """Adversarial test: GearyKhamisResult with special characters."""
    codes = ("US#1", "CA$2", "MX*3")
    gk = GearyKhamisResult(
        pi=np.ones(3),
        ppp=np.ones(3),
        real_gdp=np.array([100.0, 200.0, 300.0]),
        nominal_gdp=np.array([100.0, 200.0, 300.0]),
        gdp_growth=np.array([0.01, -0.02, 0.05]),
        country_codes=codes,
    )
    typ_det = gk.to_typst(detailed=True)
    info_det = validate_typst_table(typ_det)
    assert info_det["n_rows"] == 4
    assert r"US\#1" in typ_det
    assert r"CA\$2" in typ_det
    assert r"MX\*3" in typ_det

    typ_agg = gk.to_typst(detailed=False)
    info_agg = validate_typst_table(typ_agg)
    assert info_agg["n_rows"] == 9


def test_scenario_batch_to_typst_special_characters():
    """Adversarial test: ScenarioBatchResult with special characters in scenario and country names."""
    codes = ("US#1", "CA$2")
    s_codes = ("Sec#*", "Sec$_[]")
    eq = _make_adversarial_equilibrium(country_codes=codes, sector_codes=s_codes)
    gk = GearyKhamisResult(
        pi=np.ones(3),
        ppp=np.ones(2),
        real_gdp=np.array([100.0, 200.0]),
        nominal_gdp=np.array([100.0, 200.0]),
        gdp_growth=np.array([0.0, 0.0]),
        country_codes=codes,
    )
    batch = ScenarioBatchResult(
        scenarios={"base#0": eq, "shock$1*": eq},
        geary_khamis={"base#0": gk, "shock$1*": gk},
        baseline_scenario="base#0",
        country_codes=codes,
        sector_codes=s_codes,
    )
    typ_sel = batch.to_typst(table_type="selected")
    validate_typst_table(typ_sel)
    assert r"shock\$1\*" in typ_sel

    typ_mean = batch.to_typst(table_type="mean")
    validate_typst_table(typ_mean)
    assert r"shock\$1\*" in typ_mean


def test_retaliation_game_to_typst_special_characters():
    """Adversarial test: RetaliationGameResult with special characters in strategic player names."""
    codes = ("US#1", "CA$2")
    eq = _make_adversarial_equilibrium(country_codes=codes)
    res = RetaliationGameResult(
        scenario_name="retaliation_test#1",
        equilibrium=eq,
        initial_scenario=None,
        tau_final=np.ones((2, 2, 2, 2)),
        tau_fd_final=np.ones((2, 2, 3, 2)),
        strategic_players=("US#1", "CA$2"),
        partner_tariffs={"US#1": {"Sec#*": 0.1}, "CA$2": {"Sec#*": 0.15}},
        partner_duties_collected={"US#1": 50.0, "CA$2": 75.0},
        us_duties_collected={"US#1": 45.0, "CA$2": 60.0},
        outer_iterations=3,
        converged=True,
        outer_error=1e-5,
    )
    typ = res.to_typst()
    info = validate_typst_table(typ)
    assert info["n_rows"] == 3
    assert r"US\#1" in typ
    assert r"CA\$2" in typ


def test_jcurve_dynamic_to_typst_special_characters():
    """Adversarial test: JCurveDynamicResult with special characters."""
    eq = _make_adversarial_equilibrium()
    res = JCurveDynamicResult(
        scenario_name="jcurve#1_$*<test>",
        quarters=(0.0, 1.0, 2.0),
        sigma_path=(1.0, 2.0, 4.0),
        t_star=1.5,
        half_life=4.6,
        trade_balance_path=np.array([-10.0, -15.0, 5.0]),
        us_exports_path=np.array([100.0, 105.0, 120.0]),
        us_imports_path=np.array([110.0, 120.0, 115.0]),
        us_gdp_path=np.array([1000.0, 995.0, 1010.0]),
        us_cpi_path=np.array([1.0, 1.02, 1.01]),
        equilibria=(eq, eq, eq),
    )
    typ = res.to_typst()
    info = validate_typst_table(typ)
    assert info["n_rows"] == 4  # 1 header + 3 quarters


def test_revenue_recycling_to_typst_special_characters():
    """Adversarial test: RevenueRecyclingResult formatting."""
    eq = _make_adversarial_equilibrium()
    res = RevenueRecyclingResult(
        scenario_name="recycling#1_$*<test>",
        closure="lump_sum",
        equilibrium=eq,
        tariff_revenue=100.0,
        household_transfer=100.0,
        factor_tax_cut=0.0,
        subsidy_rate=0.0,
        welfare_decomposition={
            "terms_of_trade": 10.0,
            "deadweight_loss": -5.0,
            "fiscal_dividend": 2.0,
            "net_welfare_change": 7.0,
        },
    )
    typ = res.to_typst()
    info = validate_typst_table(typ)
    assert info["n_rows"] == 11  # 1 header + 10 metrics


def test_capacity_bottleneck_to_typst_special_characters():
    """Adversarial test: CapacityBottleneckResult with special characters in sector names."""
    eq = _make_adversarial_equilibrium()
    sectors = ("Sec#1", "Sec$2*", "Sec[3]")
    res = CapacityBottleneckResult(
        scenario_name="bottleneck#1",
        equilibrium=eq,
        capacity_margins={s: 0.05 for s in sectors},
        capacity_limits={s: 105.0 for s in sectors},
        output_levels={s: 102.0 for s in sectors},
        capacity_utilization={s: 0.97 for s in sectors},
        price_escalation={s: 3.5 for s in sectors},
        penalty_multipliers={s: 0.02 for s in sectors},
    )
    typ = res.to_typst()
    info = validate_typst_table(typ)
    assert info["n_rows"] == 4
    assert r"Sec\#1" in typ
    assert r"Sec\$2\*" in typ
    assert r"Sec\[3\]" in typ


def test_optimal_tariff_to_typst_special_characters():
    """Adversarial test: OptimalTariffResult with special characters in country code."""
    eq = _make_adversarial_equilibrium()
    res = OptimalTariffResult(
        country_code="US#1_$*",
        optimal_tariff_rate=0.15,
        welfare_gain_pct=1.2,
        baseline_welfare=100.0,
        optimal_welfare=101.2,
        welfare_metric="geary_khamis",
        terms_of_trade_initial=1.0,
        terms_of_trade_optimal=1.04,
        tariff_grid=np.linspace(0.0, 0.3, 5),
        welfare_curve=np.linspace(100.0, 101.2, 5),
        equilibrium=eq,
    )
    typ = res.to_typst()
    info = validate_typst_table(typ)
    assert info["n_rows"] == 11  # 1 header + 10 metrics
    assert r"US\#1\_\$\*" in typ


def test_nash_tariff_to_typst_special_characters():
    """Adversarial test: NashTariffResult with special characters in player codes."""
    eq = _make_adversarial_equilibrium()
    players = ("US#1", "CN$2")
    res = NashTariffResult(
        strategic_players=players,
        nash_tariffs={p: 0.15 for p in players},
        welfare_changes_pct={p: -0.5 for p in players},
        terms_of_trade_changes_pct={p: 1.0 for p in players},
        world_welfare_change_pct=-0.6,
        outer_iterations=4,
        converged=True,
        outer_error=1e-5,
        equilibrium=eq,
        tau_nash=np.ones((2, 2, 2, 2)),
        tau_fd_nash=np.ones((2, 2, 3, 2)),
    )
    typ = res.to_typst()
    info = validate_typst_table(typ)
    assert info["n_rows"] == 3
    assert r"US\#1" in typ
    assert r"CN\$2" in typ


def test_welfare_payoff_matrix_to_typst_special_characters():
    """Adversarial test: WelfarePayoffMatrixResult with special characters in player and strategy names."""
    eq = _make_adversarial_equilibrium()
    players = ("US#1", "CN$2")
    strategies = ("Coop#0", "Defect$1*")
    res = WelfarePayoffMatrixResult(
        players=players,
        strategies=strategies,
        payoff_matrix=np.array([[[0.0, 0.0], [-1.0, 0.5]], [[0.5, -1.0], [-0.5, -0.5]]]),
        scenarios={"CC": eq, "CD": eq, "DC": eq, "DD": eq},
        summary_df=pd.DataFrame({
            "Player": [players[0], players[1]],
            strategies[0]: [0.0, 0.0],
            strategies[1]: [-0.5, -0.5],
        }),
    )
    typ = res.to_typst()
    info = validate_typst_table(typ)
    assert info["n_rows"] == 3
    assert r"US\#1" in typ
    assert r"CN\$2" in typ
    assert r"Coop\#0" in typ
    assert r"Defect\$1\*" in typ


def test_bvar_sv_and_score_diagnostics_to_typst_special_characters():
    """Adversarial test: BVAR_SVForecast and ScoreDiagnosticsResult with special variable names."""
    # BVAR_SVForecast
    adv_names = ["inf#l_ation", "rate$*_1"]
    forecast = BVAR_SVForecast(
        paths=np.ones((10, 3, 2)),
        h_paths=np.zeros((10, 3, 2)),
        index=pd.RangeIndex(3),
        names=adv_names,
        history=pd.DataFrame(np.ones((5, 2)), columns=adv_names),
        ci=0.9,
    )
    typ_bvar = forecast.to_typst()
    info_bvar = validate_typst_table(typ_bvar)
    assert info_bvar["n_rows"] == 7  # 1 header + 3 horizons * 2 variables
    assert r"inf\#l\_ation" in typ_bvar
    assert r"rate\$\*\_1" in typ_bvar

    # ScoreDiagnosticsResult
    res = ScoreDiagnosticsResult(
        loglik=-50.0,
        gradient=np.array([0.01, -0.02]),
        param_names=("alpha#1", "beta$2*"),
        elapsed_sec=0.005,
    )
    typ_score = res.to_typst()
    info_score = validate_typst_table(typ_score)
    assert info_score["n_rows"] == 3
    assert r"alpha\#1" in typ_score
    assert r"beta\$2\*" in typ_score


# ===========================================================================
# 2. Challenge Extreme Numbers (NaN, Inf, -Inf, 1e308, 1e-308)
# ===========================================================================

@pytest.mark.parametrize("extreme_val", [np.nan, np.inf, -np.inf, 1e308, 1e-308, 0.0, -0.0])
def test_trade_results_extreme_numbers(extreme_val: float):
    """Stress-test trade result classes when populated with extreme numerical values."""
    codes = ("CAN", "USA")
    eq = _make_adversarial_equilibrium(values=extreme_val)
    typ = eq.to_typst(detailed=True)
    validate_typst_table(typ)

    typ_agg = eq.to_typst(detailed=False)
    validate_typst_table(typ_agg)

    # GearyKhamisResult with extreme numbers
    gk = GearyKhamisResult(
        pi=np.full(3, extreme_val),
        ppp=np.full(2, extreme_val),
        real_gdp=np.full(2, extreme_val),
        nominal_gdp=np.full(2, extreme_val),
        gdp_growth=np.full(2, extreme_val),
        country_codes=codes,
        world_nominal_gdp=extreme_val,
        world_real_gdp=extreme_val,
    )
    typ_gk = gk.to_typst(detailed=True)
    validate_typst_table(typ_gk)


# ===========================================================================
# 3. Challenge Single-Country / Single-Sector / Minimal Boundary Dimensions
# ===========================================================================

def test_single_country_single_sector_boundary():
    """Boundary test: nc=1, ns=1 minimal world model."""
    codes = ("SOLO#1",)
    s_codes = ("SECT$1",)
    
    # 1 country, 1 sector calibration
    calib = TradeCalibrationResult(
        a=np.ones((1, 1, 1)),
        afd=np.ones((1, 3, 1)),
        alpha=np.full((1, 1, 1), 0.33),
        beta=np.ones((1, 1, 1)),
        k_endow=np.ones((1, 1)),
        l_endow=np.ones((1, 1)),
        invforT=np.zeros((1, 1)),
        tax=np.zeros((1, 1, 1)),
        ytot=np.ones((1, 1, 1)),
        n_countries=1,
        n_sectors=1,
        country_codes=codes,
        sector_codes=s_codes,
    )
    typ_det = calib.to_typst(detailed=True)
    info_det = validate_typst_table(typ_det)
    assert info_det["n_rows"] == 2  # 1 header + 1 country
    assert r"SOLO\#1" in typ_det

    # 1 country equilibrium
    eq = _make_adversarial_equilibrium(country_codes=codes, sector_codes=s_codes)
    typ_eq = eq.to_typst(detailed=True)
    info_eq = validate_typst_table(typ_eq)
    assert info_eq["n_rows"] == 2

    # 1 country Geary-Khamis
    gk = GearyKhamisResult(
        pi=np.ones(1),
        ppp=np.ones(1),
        real_gdp=np.array([500.0]),
        nominal_gdp=np.array([500.0]),
        gdp_growth=np.array([0.0]),
        country_codes=codes,
    )
    typ_gk = gk.to_typst(detailed=True)
    info_gk = validate_typst_table(typ_gk)
    assert info_gk["n_rows"] == 2

    # Single-quarter J-curve
    jcurve = JCurveDynamicResult(
        scenario_name="single_q",
        quarters=(0.0,),
        sigma_path=(1.0,),
        t_star=0.0,
        half_life=1.0,
        trade_balance_path=np.array([0.0]),
        us_exports_path=np.array([100.0]),
        us_imports_path=np.array([100.0]),
        us_gdp_path=np.array([1000.0]),
        us_cpi_path=np.array([1.0]),
        equilibria=(eq,),
    )
    typ_jc = jcurve.to_typst()
    info_jc = validate_typst_table(typ_jc)
    assert info_jc["n_rows"] == 2  # 1 header + 1 quarter

    # Single-sector bottleneck
    bottleneck = CapacityBottleneckResult(
        scenario_name="single_sec",
        equilibrium=eq,
        capacity_margins={"S01#": 0.1},
        capacity_limits={"S01#": 100.0},
        output_levels={"S01#": 95.0},
        capacity_utilization={"S01#": 0.95},
        price_escalation={"S01#": 1.0},
        penalty_multipliers={"S01#": 0.0},
    )
    typ_bn = bottleneck.to_typst()
    info_bn = validate_typst_table(typ_bn)
    assert info_bn["n_rows"] == 2


# ===========================================================================
# 4. Challenge VFISolution Presentation Methods (1D, 2D, None Grids & Headless)
# ===========================================================================

@pytest.mark.parametrize(
    "grid_config",
    [
        ("1d_1d", np.linspace(0.1, 5.0, 10), np.array([0.8, 1.2])),
        ("none_none", None, None),
        ("1d_none", np.linspace(0.1, 5.0, 10), None),
        ("none_1d", None, np.array([0.8, 1.2])),
        ("2d_2d", np.ones((10, 2)), np.ones((2, 2))),  # multi-dimensional arrays
        ("mismatched_length", np.linspace(0.1, 5.0, 4), np.array([0.8, 1.0, 1.2])),  # len(a) != 10
    ],
)
def test_vfi_solution_presentation_grid_combinations(grid_config: tuple[str, Any, Any]):
    """Stress-test VFISolution presentation methods across all grid variations."""
    label, a_grid, z_grid = grid_config
    na, nz = 10, 2
    sol = VFISolution(
        V=np.ones((na, nz)) * 2.5,
        policy_aprime=np.zeros((na, nz), dtype=int),
        policy_d=None,
        n_iter=15,
        sup_norm=1e-7,
        backend="numpy",
        endo_shape=(na,),
        a_grid=a_grid,
        z_grid=z_grid,
    )

    # 1. .summary()
    sm = sol.summary()
    assert isinstance(sm, pd.DataFrame)
    assert sm.loc["Total State Space", "Value"] == str(na * nz)

    # 2. .to_frame()
    df = sol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == na * nz
    assert "V" in df.columns
    assert "policy_aprime" in df.columns

    # 3. Export quintet
    md = sol.to_markdown()
    assert isinstance(md, str) and "|" in md
    tex = sol.to_latex()
    assert isinstance(tex, str) and "\\begin{tabular}" in tex
    typ = sol.to_typst()
    validate_typst_table(typ)

    # 4. Headless .plot() safety: verify plt.show() is NEVER called when show=False
    called_show = False

    def fake_show(*args, **kwargs):
        nonlocal called_show
        called_show = True

    orig_show = plt.show
    plt.show = fake_show
    try:
        # Default ax=None
        fig = sol.plot(show=False)
        assert not called_show, "plt.show() was invoked despite show=False!"
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

        # Single axis passed
        fig_single, ax_single = plt.subplots(1, 1)
        res_ax = sol.plot(ax=ax_single, show=False)
        assert not called_show
        assert res_ax is ax_single
        plt.close(fig_single)

        # Tuple of 2 axes passed
        fig_pair, (ax1, ax2) = plt.subplots(1, 2)
        res_pair = sol.plot(ax=(ax1, ax2), show=False)
        assert not called_show
        assert res_pair == (ax1, ax2)
        plt.close(fig_pair)

        # Array of 2 axes passed
        fig_arr, axes_arr = plt.subplots(1, 2)
        res_arr = sol.plot(ax=axes_arr, show=False)
        assert not called_show
        assert np.array_equal(res_arr, axes_arr)
        plt.close(fig_arr)

        # When show=True, it SHOULD call show
        sol.plot(show=True)
        assert called_show, "plt.show() was NOT invoked when show=True"
    finally:
        plt.show = orig_show


def test_vfi_solution_with_discrete_choice():
    """Verify VFISolution when discrete choice policy is present."""
    na, nz = 5, 2
    sol = VFISolution(
        V=np.ones((na, nz)),
        policy_aprime=np.zeros((na, nz), dtype=int),
        policy_d=np.ones((na, nz), dtype=int),
        n_iter=5,
        sup_norm=1e-5,
        backend="numpy",
        endo_shape=(na,),
        a_grid=np.linspace(0.1, 1.0, na),
        z_grid=np.array([0.5, 1.5]),
    )
    df = sol.to_frame()
    assert "policy_d" in df.columns
    assert len(df) == 10
    sm = sol.summary()
    assert sm.loc["Discrete Choice Included", "Value"] == "True"


# ===========================================================================
# 5. Challenge HJBSolution Dict Protocol, Frozen Immutability & Headless Plot
# ===========================================================================

def test_hjb_solution_frozen_immutability():
    """Adversarial test: verify HJBSolution is strictly immutable."""
    sol = solve_hjb_achdou(Na=20, max_iter=10)
    assert isinstance(sol, HJBSolution)

    # Attempt attribute mutation
    with pytest.raises(FrozenInstanceError):
        sol.V = np.zeros((20, 2))

    with pytest.raises(FrozenInstanceError):
        sol.n_iter = 999

    with pytest.raises(FrozenInstanceError):
        sol.converged = False

    # Attempt attribute deletion
    with pytest.raises(FrozenInstanceError):
        del sol.V

    with pytest.raises(FrozenInstanceError):
        del sol.converged

    # Attempt adding new arbitrary attribute
    with pytest.raises(FrozenInstanceError):
        sol.extra_attribute = "adversarial"


def test_hjb_solution_dict_mapping_protocol():
    """Comprehensive test of the dictionary mapping protocol on HJBSolution."""
    sol = solve_hjb_achdou(Na=20, max_iter=10)
    expected_keys = [
        "V",
        "c_policy",
        "s_drift",
        "a_grid",
        "e_grid",
        "n_iter",
        "elapsed",
        "r_rate",
        "w_rate",
        "rho_val",
        "gamma_r",
        "converged",
    ]

    # 1. __getitem__ for valid keys
    for k in expected_keys:
        val = sol[k]
        assert val is getattr(sol, k), f"Mismatch for key '{k}'"

    # 2. __getitem__ raises KeyError on nonexistent / edge-case keys
    with pytest.raises(KeyError):
        _ = sol["nonexistent_key"]

    with pytest.raises(KeyError):
        _ = sol[""]

    with pytest.raises(KeyError):
        _ = sol["_private_field"]

    # 3. __contains__
    for k in expected_keys:
        assert k in sol, f"Key '{k}' not found in sol"
    assert "nonexistent_key" not in sol
    assert "" not in sol
    assert 123 not in sol
    assert None not in sol
    assert () not in sol

    # 4. keys(), values(), items()
    assert sol.keys() == expected_keys
    vals = sol.values()
    assert len(vals) == len(expected_keys)
    items = sol.items()
    assert len(items) == len(expected_keys)
    for (k, v), exp_k in zip(items, expected_keys):
        assert k == exp_k
        assert v is getattr(sol, exp_k)

    # 5. dict(sol.items()) conversion
    as_dict = dict(sol.items())
    assert isinstance(as_dict, dict)
    assert set(as_dict.keys()) == set(expected_keys)

    # 6. get() method
    assert sol.get("V") is sol.V
    assert sol.get("nonexistent_key") is None
    assert sol.get("nonexistent_key", "default_val") == "default_val"


def test_hjb_solution_presentation_and_headless_plot():
    """Test HJBSolution tabular presentations and headless .plot()."""
    sol = solve_hjb_achdou(Na=20, max_iter=10)

    # Summary
    sm = sol.summary()
    assert isinstance(sm, pd.DataFrame)
    assert "Asset Grid Points (Na)" in sm.index

    # to_frame
    df = sol.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 40  # 20 * 2
    assert set(df.columns) >= {"asset_a", "prod_e", "value_V", "consumption_c", "savings_drift_s"}

    # Typst / LaTeX / Markdown
    typ = sol.to_typst()
    validate_typst_table(typ)
    assert "\\begin{tabular}" in sol.to_latex()
    assert "|" in sol.to_markdown()

    # Headless plot check
    called_show = False

    def fake_show(*args, **kwargs):
        nonlocal called_show
        called_show = True

    orig_show = plt.show
    plt.show = fake_show
    try:
        # Default ax=None (creates 1x3)
        fig = sol.plot(show=False)
        assert not called_show, "plt.show() was invoked despite show=False!"
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(fig.axes) == 3
        plt.close(fig)

        # Single axis passed
        fig_single, ax_single = plt.subplots(1, 1)
        res_single = sol.plot(ax=ax_single, show=False)
        assert not called_show
        assert res_single is ax_single
        plt.close(fig_single)

        # 3 axes passed
        fig_3, (ax1, ax2, ax3) = plt.subplots(1, 3)
        res_3 = sol.plot(ax=(ax1, ax2, ax3), show=False)
        assert not called_show
        assert res_3 == (ax1, ax2, ax3)
        plt.close(fig_3)

        # show=True triggers show
        sol.plot(show=True)
        assert called_show, "plt.show() was not invoked when show=True!"
    finally:
        plt.show = orig_show
