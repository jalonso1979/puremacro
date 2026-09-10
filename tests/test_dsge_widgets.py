"""Test suite for live interactive parameter slider widgets (puremacro.dsge.widgets)."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import build_dynare, load_mod, LinearModel
from puremacro.dsge.build import ModelError
from puremacro.dsge.widgets import interactive_irf, InteractiveIRFResult


# ==============================================================================
# Model Fixtures
# ==============================================================================

def _nk_equations(lead, curr, lag, shocks, p):
    """Textbook 3-equation New Keynesian model with AR(1) cost-push and monetary shocks.

    1. IS curve: y_t = E_t y_{t+1} - (1/sigma) * (r_t - E_t pi_{t+1})
    2. NKPC: pi_t = beta * E_t pi_{t+1} + kappa * y_t + u_t
    3. Taylor rule: r_t = phi_pi * pi_t + phi_y * y_t + eps_r
    4. Cost-push shock AR(1): u_t = rho_u * lag.u + eps_u
    """
    eq_is = lead.y - curr.y - (curr.r - lead.pi) / p.sigma
    eq_pc = p.beta * lead.pi + p.kappa * curr.y + curr.u - curr.pi
    eq_tr = p.phi_pi * curr.pi + p.phi_y * curr.y + shocks.eps_r - curr.r
    eq_u = p.rho_u * lag.u + shocks.eps_u - curr.u
    return [eq_is, eq_pc, eq_tr, eq_u]


def _build_nk_model(
    *,
    beta: float = 0.99,
    sigma: float = 1.0,
    kappa: float = 0.05,
    phi_pi: float = 1.5,
    phi_y: float = 0.1,
    rho_u: float = 0.5,
) -> LinearModel:
    return build_dynare(
        _nk_equations,
        variables=["y", "pi", "r", "u"],
        states=["u"],
        shocks=["eps_r", "eps_u"],
        params={
            "beta": beta,
            "sigma": sigma,
            "kappa": kappa,
            "phi_pi": phi_pi,
            "phi_y": phi_y,
            "rho_u": rho_u,
        },
        steady_state={"y": 0.0, "pi": 0.0, "r": 0.0, "u": 0.0},
        strict=True,
    )


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_interactive_irf_instantiation():
    """Test widget instantiation, tuple unpacking, types, and counts."""
    model = _build_nk_model()
    res = model.interactive_irf(
        parameters=["sigma", "kappa", "phi_pi"],
        shocks=["eps_r"],
        horizon=20,
    )
    assert isinstance(res, InteractiveIRFResult)
    assert res.fig is not None
    assert isinstance(res.fig, matplotlib.figure.Figure)
    assert len(res.sliders) == 3
    assert set(res.sliders.keys()) == {"sigma", "kappa", "phi_pi"}
    assert len(res.lines) == len(res.variables)
    assert len(res.baseline_lines) == len(res.variables)
    assert res.model.is_determinate

    # Tuple unpacking support
    fig, sliders = model.interactive_irf(
        parameters=["sigma", "kappa", "phi_pi"],
        shocks=["eps_r"],
        horizon=20,
    )
    assert fig is not None
    assert isinstance(fig, matplotlib.figure.Figure)
    assert len(sliders) == 3
    assert set(sliders.keys()) == {"sigma", "kappa", "phi_pi"}
    plt.close("all")


def test_slider_change_callback_updates_irf():
    """Test that adjusting a slider dynamically updates Line2D data with < 15ms latency."""
    model = _build_nk_model()
    res = model.interactive_irf(
        parameters=["sigma", "kappa", "phi_pi"],
        shocks=["eps_r"],
        horizon=20,
    )
    y0 = res.lines[("y", "eps_r")].get_ydata().copy()
    res.set_value("phi_pi", 2.5)
    y1 = res.lines[("y", "eps_r")].get_ydata()

    assert not np.allclose(y0, y1)
    assert res.last_latency_ms < 15.0
    assert res.n_updates >= 1
    assert res.model.is_determinate
    plt.close("all")


def test_blanchard_kahn_violation_handling():
    """Test graceful NaN display and status warning banner on Blanchard-Kahn violation."""
    model = _build_nk_model()
    res = model.interactive_irf(
        parameters=["sigma", "kappa", "phi_pi"],
        shocks=["eps_r"],
        horizon=20,
    )
    # Drive phi_pi below 1.0 (triggers Blanchard-Kahn indeterminacy)
    res.set_value("phi_pi", 0.5)
    assert not res.model.is_determinate
    y_viol = res.lines[("y", "eps_r")].get_ydata()
    assert np.isnan(y_viol).all()
    assert len(res.status_text.get_text()) > 0
    assert "Blanchard-Kahn" in res.status_text.get_text()

    # Restore parameter back to determinate region
    res.set_value("phi_pi", 1.5)
    assert res.model.is_determinate
    y_restored = res.lines[("y", "eps_r")].get_ydata()
    assert not np.isnan(y_restored).any()
    assert res.status_text.get_text() == ""
    plt.close("all")


def test_reset_functionality():
    """Test that reset() restores all sliders and models to initial baseline calibration."""
    model = _build_nk_model()
    res = model.interactive_irf(
        parameters=["sigma", "kappa", "phi_pi"],
        shocks=["eps_r"],
        horizon=20,
    )
    res.update({"phi_pi": 2.8, "kappa": 0.15})
    assert res.get_values()["phi_pi"] == pytest.approx(2.8)
    assert res.get_values()["kappa"] == pytest.approx(0.15)

    res.reset()
    assert res.get_values()["phi_pi"] == pytest.approx(1.5)
    assert res.get_values()["kappa"] == pytest.approx(0.05)
    assert res.get_values()["sigma"] == pytest.approx(1.0)
    plt.close("all")


def test_presentation_contract():
    """Test puremacro presentation interfaces (.summary, .to_markdown, .to_latex, .to_typst, .plot)."""
    model = _build_nk_model()
    res = model.interactive_irf(
        parameters=["sigma", "kappa", "phi_pi"],
        shocks=["eps_r"],
        horizon=20,
    )
    assert len(res.summary()) > 0
    assert "INTERACTIVE IMPULSE RESPONSE EXPLORATION" in res.summary()
    assert len(res.to_markdown()) > 0
    assert len(res.to_latex()) > 0
    assert len(res.to_typst()) > 0
    assert res.plot() is res.fig
    df = res.to_frame()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 21  # 0..20
    plt.close("all")


def test_custom_bounds_and_steps():
    """Test custom slider bounds, initial values, and step increments."""
    model = _build_nk_model()
    res = model.interactive_irf(
        parameters={"phi_pi": (1.1, 4.0, 2.0), "kappa": (0.01, 0.3)},
        shocks=["eps_r"],
        horizon=20,
        param_steps={"phi_pi": 0.05},
    )
    assert res.sliders["phi_pi"].valmin == pytest.approx(1.1)
    assert res.sliders["phi_pi"].valmax == pytest.approx(4.0)
    assert res.sliders["phi_pi"].val == pytest.approx(2.0)
    assert res.sliders["phi_pi"].valstep == pytest.approx(0.05)

    assert res.sliders["kappa"].valmin == pytest.approx(0.01)
    assert res.sliders["kappa"].valmax == pytest.approx(0.3)
    assert res.sliders["kappa"].val == pytest.approx(0.05)
    plt.close("all")


def test_multi_shock_and_multi_variable():
    """Test interactive exploration across multiple shocks and variables."""
    model = _build_nk_model()
    res = model.interactive_irf(
        parameters=["phi_pi", "kappa"],
        shocks=["eps_r", "eps_u"],
        variables=["y", "pi", "r"],
        horizon=15,
    )
    assert len(res.lines) == 6
    for var in ["y", "pi", "r"]:
        for sh in ["eps_r", "eps_u"]:
            assert (var, sh) in res.lines
            assert (var, sh) in res.axes
            line = res.lines[(var, sh)]
            assert len(line.get_ydata()) == 16
            assert not np.isnan(line.get_ydata()).any()
    plt.close("all")


def test_sw07_large_model_interactive():
    """Test interactive exploration on large Smets-Wouters (2007) model (< 15ms latency)."""
    mod_path = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
    if not mod_path.is_file():
        pytest.skip("sw07_pfeifer.mod reference file not found")
    model = load_mod(mod_path)
    res = model.interactive_irf(
        parameters=["crr", "crhopinf"],
        shocks=["ea"],
        horizon=20,
    )
    assert len(res.sliders) == 2
    res.set_value("crr", 0.85)  # warm-up
    res.set_value("crr", 0.88)
    assert res.last_latency_ms < 15.0
    assert res.n_updates >= 1
    plt.close("all")


def test_headless_execution_and_cleanup():
    """Test headless execution under Agg backend and resource cleanup via disconnect()."""
    assert matplotlib.get_backend().lower() == "agg"
    model = _build_nk_model()
    res = model.interactive_irf(
        parameters=["sigma", "kappa"],
        shocks=["eps_r"],
        horizon=10,
    )
    assert getattr(res.fig, "_interactive_widget", None) is res
    res.disconnect()
    for s in res.sliders.values():
        assert not s.eventson
    if res.reset_button is not None:
        assert not res.reset_button.eventson
    plt.close(res.fig)
    plt.close("all")


def test_invalid_inputs():
    """Test defensive error handling for invalid parameters, shocks, and negative horizons."""
    model = _build_nk_model()
    # 1. Invalid parameter name
    with pytest.raises(ValueError, match="Unknown parameter"):
        model.interactive_irf(parameters=["nonexistent_param"])

    # 2. Invalid shock name
    with pytest.raises((ModelError, ValueError)):
        model.interactive_irf(shocks=["nonexistent_shock"])

    # 3. Negative horizon
    with pytest.raises(ValueError, match="horizon must be non-negative"):
        model.interactive_irf(horizon=-5)

    # 4. Unknown parameter in set_value
    res = model.interactive_irf(parameters=["phi_pi"])
    with pytest.raises(KeyError, match="No slider"):
        res.set_value("nonexistent_param", 1.0)
    plt.close("all")


def test_flexible_registry_string_key_indexing():
    """Test string key indexing on lines and axes registries (e.g. res.lines['y'], res.axes['y'])."""
    model = _build_nk_model()
    # Single shock: variable string lookup is unique and unambiguous
    res = model.interactive_irf(
        shocks=["eps_r"],
        variables=["y", "pi"],
    )
    # String key lookup
    assert "y" in res.lines
    assert "pi" in res.lines
    assert "y" in res.axes
    assert "pi" in res.axes

    line_y = res.lines["y"]
    assert isinstance(line_y, matplotlib.lines.Line2D)
    assert line_y is res.lines[("y", "eps_r")]

    ax_y = res.axes["y"]
    assert isinstance(ax_y, matplotlib.axes.Axes)
    assert ax_y is res.axes[("y", "eps_r")]

    # Unknown key raises KeyError
    with pytest.raises(KeyError):
        _ = res.lines["nonexistent"]
    with pytest.raises(KeyError):
        _ = res.axes["nonexistent"]

    plt.close("all")

    # Multi-shock: variable string lookup is ambiguous across shocks
    res_multi = model.interactive_irf(
        shocks=["eps_r", "eps_u"],
        variables=["y"],
    )
    with pytest.raises(KeyError, match="Multiple lines match variable 'y'"):
        _ = res_multi.lines["y"]

    plt.close("all")


def test_single_string_variable_argument():
    """Test that passing a single string for variables (e.g. variables='y' or 'pi') normalizes properly."""
    model = _build_nk_model()
    # 1-character variable
    res_y = model.interactive_irf(variables="y", shocks="eps_r")
    assert ("y", "eps_r") in res_y.lines
    assert len(res_y.lines) == 1
    assert res_y.lines["y"] is not None
    plt.close("all")

    # Multi-character variable (guards against list('pi') -> ['p', 'i'] splitting bug)
    res_pi = model.interactive_irf(variables="pi", shocks="eps_r")
    assert ("pi", "eps_r") in res_pi.lines
    assert len(res_pi.lines) == 1
    assert res_pi.lines["pi"] is not None
    plt.close("all")
