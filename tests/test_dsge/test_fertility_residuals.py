"""Tests for puremacro.dsge.fertility_adj_costs.model_residuals."""
from __future__ import annotations

import warnings

import numpy as np
import pytest


def _build_z_ss_and_params():
    """Helper: the pinned calibration + its EXACT steady state (R1).

    Previously this solved the over-determined BGP and linearised at its
    least-squares endpoint — a non-equilibrium point whose residuals were
    only ``< 1.0`` and whose value flipped across LAPACK builds. R1 uses the
    exact steady state of the pinned calibration, at which the dynamic
    equations hold to machine precision on every platform.
    """
    from puremacro.dsge.fertility_adj_costs import (
        FERTILITY_PINNED_CALIBRATION, FERTILITY_SHOCK_PROCESSES,
        exact_steady_state,
    )
    params = dict(FERTILITY_PINNED_CALIBRATION)
    for shock, spec in FERTILITY_SHOCK_PROCESSES.items():
        prefix = {"ea": "a", "en": "n", "ep": "p"}[shock]
        params[f"rho{prefix}"] = spec["rho"]
        params[f"sigma{prefix}"] = spec["sigma"]
    z_ss = exact_steady_state(params)
    return z_ss, params


def test_model_residuals_zero_at_steady_state():
    """At the exact steady state (R1), model residuals are machine-zero.

    ``exact_steady_state`` solves ``model_residuals(z, z, z, 0) = 0``, so
    this is a genuine equilibrium (unlike the old over-determined-BGP
    least-squares point, whose residual was only ``< 1.0`` and drifted
    across LAPACK builds).
    """
    from puremacro.dsge.fertility_adj_costs import model_residuals
    z_ss, params = _build_z_ss_and_params()
    eps = np.zeros(3)
    res = model_residuals(z_ss, z_ss, z_ss, eps, params)
    max_abs = float(np.max(np.abs(res)))
    assert max_abs < 1e-9, (
        f"max |residual| at exact SS = {max_abs:.3e} — not a steady state"
    )


def test_model_residuals_returns_correct_shape():
    from puremacro.dsge.fertility_adj_costs import model_residuals
    z_ss, params = _build_z_ss_and_params()
    res = model_residuals(z_ss, z_ss, z_ss, np.zeros(3), params)
    assert res.shape == (12,)


def test_model_residuals_responds_to_perturbation():
    from puremacro.dsge.fertility_adj_costs import model_residuals, VAR_NAMES
    z_ss, params = _build_z_ss_and_params()
    eps = np.zeros(3)
    base = model_residuals(z_ss, z_ss, z_ss, eps, params)
    # Perturb c (consumption) by 1% at current time
    z_pert = z_ss.copy()
    z_pert[VAR_NAMES.index("c")] *= 1.01
    perturbed = model_residuals(z_ss, z_pert, z_ss, eps, params)
    assert np.max(np.abs(perturbed - base)) > 1e-6
