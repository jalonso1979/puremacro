"""Validation cases for the 77-country, 11-sector trade model on the clean OECD 2020 table.

Through 4.5.0 the gallery had no trade cases: the only bundled table was the
MATLAB-parity fixture built from a corrupted OECD export, whose economics
could not be checked against anything outside itself. The clean table bundled
in 4.6.0 (``load_icio_data(source="oecd2020")``, built by
``tools/build_icio_77c_11s.py``) can: its aggregation must conserve the world
totals of the native OECD file it was built from, its columns must balance,
the calibrated model must reproduce the base year as its own equilibrium, and
that equilibrium must agree with the solution of the author's legacy MATLAB
model on the same table (``trade_reference_solutions_oecd2020.npz``, an
external reference copied verbatim).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._model import Mechanism, Tol, ValidationCase

_NC, _NS, _NFD = 77, 11, 3
_N_IND = _NC * _NS
_SUBSYSTEM = "Trade"
_OECD = "OECD Inter-Country Input-Output tables, 2023 edition, 2020 table (2020_SML.csv, MD5 d3e0f497...)"


def _table() -> np.ndarray:
    from puremacro.trade.data import load_icio_data

    return load_icio_data(source="oecd2020")


def _manifest() -> dict:
    from puremacro.trade.data import bundled_icio_path

    return json.loads((bundled_icio_path("oecd2020").parent / "MANIFEST_OECD2020.json").read_text(encoding="utf-8"))


def _world_totals() -> dict:
    M = _table()
    va = M[_N_IND + 1, :_N_IND] + M[_N_IND + 2, :_N_IND]
    return {"world_value_added": float(va.sum()), "world_gross_output": float(M[:_N_IND].sum())}


def _native_totals() -> dict:
    t = _manifest()["source_totals"]
    return {"world_value_added": t["world_value_added"], "world_gross_output": t["world_gross_output"]}


def _column_balance() -> dict:
    """max_j |sales_j - purchases_j - taxes_j - VA_j| / sales_j over the 847 industry columns."""
    M = _table()
    sales = M[:_N_IND].sum(axis=1)
    outlays = M[:_N_IND, :_N_IND].sum(axis=0) + M[_N_IND, :_N_IND] + M[_N_IND + 1, :_N_IND] + M[_N_IND + 2, :_N_IND]
    return {"max_relative_imbalance": float(np.max(np.abs(sales - outlays) / sales)),
            "negative_value_added_cells": float(np.sum((M[_N_IND + 1, :_N_IND] + M[_N_IND + 2, :_N_IND]) <= 0))}


def _base_solution():
    from puremacro.trade import calibrate_trade_model, solve_trade_equilibrium

    calib = calibrate_trade_model(_table(), ns=_NS, nc=_NC, nfd=_NFD, validate=True)
    return solve_trade_equilibrium(calib, method="newton", tol=2.5e-3)


def _base_is_calibration_point() -> dict:
    res = _base_solution()
    return {
        "max_abs_log_price": float(np.max(np.abs(np.log(res.p_sol)))),
        "max_abs_log_wage": float(np.max(np.abs(np.log(res.w_sol)))),
        "max_abs_log_rental": float(np.max(np.abs(np.log(res.r_sol)))),
        "converged": float(res.converged),
    }


def _base_x_sol() -> dict:
    return {"xx_sol": np.asarray(_base_solution().x_sol, dtype=float).ravel()}


def _matlab_base_x_sol() -> dict:
    from puremacro.trade.data import load_reference_solution

    return {"xx_sol": np.asarray(load_reference_solution("base", source="oecd2020")["xx_sol"], dtype=float).ravel()}


CASES: list[ValidationCase] = [
    ValidationCase(
        id="trade.oecd2020_aggregation_conserves_native_world_totals",
        subsystem=_SUBSYSTEM,
        title="Clean 77x11 table: world value added and gross output equal the native OECD 2020 file's",
        title_es="Tabla limpia 77x11: el valor agregado y la producción bruta mundiales igualan los del archivo nativo OCDE 2020",
        mechanism=Mechanism.INTERNAL,
        compute=_world_totals,
        reference=_native_totals,
        tol=Tol.TIGHT,
        citation=_OECD,
        notes="The native totals (45 industries x 77 countries, 3,468 x 3,928 file) are frozen in "
              "MANIFEST_OECD2020.json by tools/build_icio_77c_11s.py; the aggregation to 11 sectors and "
              "3 final-demand categories must move cells, not create or destroy value. World value added is "
              "7.97e7 USD million; the legacy fixture's was 7.05e11.",
    ),
    ValidationCase(
        id="trade.oecd2020_columns_balance_and_value_added_is_positive",
        subsystem=_SUBSYSTEM,
        title="Clean 77x11 table: every industry column balances (sales = purchases + taxes + value added) with positive value added",
        title_es="Tabla limpia 77x11: cada columna industrial cuadra (ventas = compras + impuestos + valor agregado) con valor agregado positivo",
        mechanism=Mechanism.INTERNAL,
        compute=_column_balance,
        reference=lambda: {"max_relative_imbalance": 0.0, "negative_value_added_cells": 0.0},
        tol=Tol.TIGHT,
        citation=_OECD,
        notes="Net taxes are the accounting residual of the aggregation, so the identity holds to rounding; "
              "the case also pins that no country-sector has non-positive value added, which the "
              "corrupted legacy export could not guarantee.",
    ),
    ValidationCase(
        id="trade.oecd2020_base_equilibrium_is_the_calibration_point",
        subsystem=_SUBSYSTEM,
        title="Clean 77x11 table: the calibrated model's zero-tariff equilibrium is the base year (unit prices, wages and rentals)",
        title_es="Tabla limpia 77x11: el equilibrio sin aranceles del modelo calibrado es el año base (precios, salarios y rentas unitarios)",
        mechanism=Mechanism.INTERNAL,
        compute=_base_is_calibration_point,
        reference=lambda: {"max_abs_log_price": 0.0, "max_abs_log_wage": 0.0, "max_abs_log_rental": 0.0, "converged": 1.0},
        tol=Tol.TIGHT,
        citation="calibrate_trade_model / solve_trade_equilibrium (puremacro.trade), Newton, tol 2.5e-3",
        notes="Calibration reproduces the data (max column error 3.6e-9 in the MATLAB check), so the "
              "equilibrium system evaluated at unit prices has an L1 residual of 1.5e-7 and the solver "
              "accepts the starting point.",
    ),
    ValidationCase(
        id="trade.oecd2020_base_equilibrium_matches_matlab_reference",
        subsystem=_SUBSYSTEM,
        title="Clean 77x11 table: base equilibrium vector agrees with the legacy MATLAB model's solution on the same table",
        title_es="Tabla limpia 77x11: el vector de equilibrio base coincide con la solución del modelo MATLAB heredado sobre la misma tabla",
        mechanism=Mechanism.PACKAGE,
        compute=_base_x_sol,
        reference=_matlab_base_x_sol,
        tol=Tol.TIGHT,
        citation="MATLAB R2026a run of the author's legacy trade model (calibrar.m, ff_equi.m, ff_eval.m) on the "
                 "clean table, 2026-10-03; trade_reference_solutions_oecd2020.npz, copied verbatim",
        notes="2,001 unknowns (log prices, log outputs, log rentals, log wages, transfers, net foreign "
              "transfers). The MATLAB arrays are an external reference, not puremacro output; "
              "REFERENCE_MANIFEST_OECD2020.json records the run.",
    ),
]
