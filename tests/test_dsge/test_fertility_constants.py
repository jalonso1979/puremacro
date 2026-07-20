"""Tests for puremacro.dsge.fertility_adj_costs constants."""
from __future__ import annotations


def test_var_names_has_12_entries_in_state_then_control_order():
    from puremacro.dsge.fertility_adj_costs import VAR_NAMES
    assert len(VAR_NAMES) == 12
    # First 5 are states: a, mun, ph, k, n
    assert VAR_NAMES[:5] == ("a", "mun", "ph", "k", "n")
    # Next 7 are controls: c, y, l_w, u, i, b, l_o
    assert VAR_NAMES[5:] == ("c", "y", "l_w", "u", "i", "b", "l_o")


def test_shock_names_has_3_entries():
    from puremacro.dsge.fertility_adj_costs import SHOCK_NAMES
    assert SHOCK_NAMES == ("ea", "ep", "en")


def test_exogenous_params_have_expected_keys():
    from puremacro.dsge.fertility_adj_costs import (
        FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS,
        FERTILITY_SHOCK_PROCESSES,
    )
    assert set(FERTILITY_EXOGENOUS_PARAMS.keys()) == {
        "alpha", "nu", "phi", "g", "delta_p", "delta_n", "omega", "bara",
    }
    assert set(FERTILITY_CALIB_TARGETS.keys()) == {
        "l", "u", "depr_rate", "kid_cost_share", "k_y_ratio", "c_y_ratio",
        "n_growth",
    }
    assert set(FERTILITY_SHOCK_PROCESSES.keys()) == {"ea", "en", "ep"}
    # Spot-check a few values
    assert FERTILITY_EXOGENOUS_PARAMS["alpha"] == 0.4
    assert FERTILITY_CALIB_TARGETS["k_y_ratio"] == 2.8
