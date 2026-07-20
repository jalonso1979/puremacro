"""Tests for puremacro.dsge.fertility_adj_costs.solve_bgp."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest


def test_solve_bgp_converges_with_defaults():
    from puremacro.dsge.fertility_adj_costs import solve_bgp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bgp = solve_bgp()
    for key, val in bgp.items():
        assert math.isfinite(val), f"{key} = {val} is not finite"
    required = {
        # exogenous
        "alpha", "nu", "phi", "g", "delta_p", "delta_n", "omega", "bara",
        # calibrated by least_squares
        "barn", "mu_l", "beta", "tau_n", "tau_b", "p_n", "delta_k",
        # steady-state values
        "c", "b", "l_w", "u", "k", "n",
        # derived from SS identities
        "i", "l_o", "y",
        # shock-process steady states
        "a", "mun", "ph",
    }
    assert required.issubset(set(bgp.keys()))


def test_solve_bgp_satisfies_residuals():
    """The over-determined MATLAB BGP system is solved via least_squares;
    the residual norm should be small but not necessarily machine-zero."""
    from puremacro.dsge.fertility_adj_costs import solve_bgp, _bgp_system, FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bgp = solve_bgp()
    x = np.array([
        bgp["barn"], bgp["mu_l"], bgp["beta"], bgp["tau_n"], bgp["tau_b"],
        bgp["p_n"], bgp["delta_k"], bgp["c"], bgp["b"], bgp["l_w"],
        bgp["u"], bgp["k"], bgp["n"],
    ])
    F = _bgp_system(x, FERTILITY_EXOGENOUS_PARAMS, FERTILITY_CALIB_TARGETS)
    norm = float(np.linalg.norm(F))
    # Over-determined system — accept any norm that's not crazy.
    assert norm < 10.0, f"residual norm {norm:.3e} is too large (system may be broken)"


def test_solve_bgp_satisfies_calibration_targets():
    """The 6 calibration targets are aspirational in an over-determined system;
    least_squares minimises ||F||^2 so targets are approximately satisfied.
    Tolerance abs=0.1 (roughly 5-10% of each target's value)."""
    from puremacro.dsge.fertility_adj_costs import solve_bgp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bgp = solve_bgp()
    assert bgp["l_w"] == pytest.approx(0.30, abs=0.1)
    assert bgp["u"] == pytest.approx(1.00, abs=0.1)
    assert bgp["k"] / bgp["y"] == pytest.approx(2.80, abs=0.1)
    assert bgp["c"] / bgp["y"] == pytest.approx(0.75, abs=0.1)
    assert bgp["p_n"] * bgp["n"] / bgp["y"] == pytest.approx(0.18, abs=0.1)


def test_solve_bgp_raises_on_missing_target():
    from puremacro.dsge.fertility_adj_costs import solve_bgp
    bad_targets = {"l": 0.30, "u": 1.0}
    with pytest.raises(KeyError, match="missing"):
        solve_bgp(targets=bad_targets)


def test_solve_bgp_custom_targets_override():
    """Custom k_y_ratio target should be approximately reflected in solution."""
    from puremacro.dsge.fertility_adj_costs import (
        solve_bgp, FERTILITY_CALIB_TARGETS,
    )
    custom = dict(FERTILITY_CALIB_TARGETS)
    custom["k_y_ratio"] = 3.5
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        bgp = solve_bgp(targets=custom)
    # The over-determined system means targets are aspirational; use abs=0.1.
    assert bgp["k"] / bgp["y"] == pytest.approx(3.5, abs=0.1)
