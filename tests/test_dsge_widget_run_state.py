"""Regression tests for displayed/exported state after edits and failed solves."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from puremacro.dsge import load_mod, ModelError, SteadyStateError
import puremacro.dsge.widgets as widgets

SOURCE = "var x; varexo e; parameters rho other; rho=.8; other=1; model; x=rho*x(-1)+other*e; end; initval; x=0; end;"


@pytest.fixture
def ui():
    result = load_mod(SOURCE).interactive_irf(parameters=["rho", "other"], shocks="e", horizon=3)
    yield result
    plt.close("all")


@pytest.mark.parametrize("error,category", [(SteadyStateError("bad steady state"), "steady_state"),
                                           (ModelError("bad derivative"), "model"),
                                           (FloatingPointError("invalid domain"), "domain")])
def test_failure_never_exports_previous_result(ui, monkeypatch, error, category):
    previous = ui.last_success_model
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(widgets, "_fast_resolve", fail)
    ui.set_value("rho", .6)
    assert ui.run_state == "failed" and not ui.is_current
    assert ui.last_success_model is previous
    assert ui.to_frame().isna().all().all()
    assert all(np.isnan(line.get_ydata()).all() for line in ui.lines.values())
    assert ui.failure.category == category
    assert "Blanchard-Kahn" not in ui.status_text.get_text()
    assert "No current solved result" in ui.summary()
    assert ui.capture_run().status == "failed" and ui.capture_run().table_names == ()


def test_failed_initial_override_is_visible_and_never_uses_baseline(monkeypatch):
    model = load_mod(SOURCE)
    def fail(*args, **kwargs):
        raise SteadyStateError("initial override failed")
    monkeypatch.setattr(widgets, "_fast_resolve", fail)
    ui = model.interactive_irf(parameters={"rho": (.1, .99, .6)}, shocks="e")
    assert ui.run_state == "failed" and ui.failure.category == "steady_state"
    assert ui.to_frame().isna().all().all()
    assert ui.accepted_parameters["rho"] == .8
    assert ui.get_values()["rho"] == .6
    plt.close("all")


def test_batch_update_and_reset_solve_once_and_validate_before_editing(ui, monkeypatch):
    calls = []
    original = widgets._fast_resolve
    def counted(*args, **kwargs):
        calls.append(args[1].copy())
        return original(*args, **kwargs)
    monkeypatch.setattr(widgets, "_fast_resolve", counted)
    ui.update({"rho": .6, "other": 1.2})
    assert calls == [{"rho": .6, "other": 1.2}]
    np.testing.assert_allclose(ui.to_frame().x, 1.2 * .6 ** np.arange(4), atol=1e-13)
    before = ui.get_values()
    with pytest.raises(KeyError):
        ui.update({"rho": .4, "absent": 1.})
    assert ui.get_values() == before and len(calls) == 1
    with pytest.raises(ValueError, match="finite"):
        ui.update({"rho": .4, "other": np.nan})
    assert ui.get_values() == before
    ui.reset()
    assert len(calls) == 2 and ui.is_current


@pytest.mark.pyodide_smoke
def test_manual_apply_has_no_stale_exports_and_one_real_solve():
    ui = load_mod(SOURCE).interactive_irf(parameters=["rho", "other"], shocks="e", horizon=3, auto_solve=False)
    assert ui.is_current and ui.apply_button is not None
    ui.update({"rho": .6, "other": 1.2})
    assert ui.n_updates == 0 and ui.run_state == "edited" and not ui.is_current
    assert ui.to_frame().isna().all().all()
    assert ui.capture_run().status == "edited"
    ui.apply()
    assert ui.n_updates == 1 and ui.is_current
    np.testing.assert_allclose(ui.to_frame().x, 1.2 * .6 ** np.arange(4), atol=1e-13)
    snap = ui.capture_run(label="Accepted")
    ui.set_value("rho", .4)
    assert snap.status == "succeeded"
    np.testing.assert_allclose(snap.table().x, 1.2 * .6 ** np.arange(4), atol=1e-13)
    plt.close("all")


def test_export_freshness_survives_external_disabled_callbacks(ui):
    ui.sliders["rho"].eventson = False
    ui.sliders["rho"].set_val(.3)
    assert not ui.is_current and ui.to_frame().isna().all().all()
    assert "No current solved result" in ui.summary()


def test_initial_step_rounding_solves_the_displayed_calibration():
    ui = load_mod(SOURCE).interactive_irf(parameters={"rho": (.1, .99, .8, .3)}, shocks="e", horizon=3)
    assert ui.is_current and ui.get_values()["rho"] == pytest.approx(.7)
    np.testing.assert_allclose(ui.to_frame().x, .7 ** np.arange(4), atol=1e-13)
    plt.close("all")


def test_initial_bounds_reject_silent_clipping_and_disconnect_covers_apply():
    with pytest.raises(ValueError, match="outside"):
        load_mod(SOURCE).interactive_irf(parameters={"rho": (.1, .5)}, shocks="e")
    ui = load_mod(SOURCE).interactive_irf(parameters=["rho"], shocks="e", auto_solve=False)
    ui.disconnect()
    assert not ui.apply_button.eventson and not ui.reset_button.eventson
    plt.close("all")


def test_model_mutation_cannot_recompute_an_exported_path(ui):
    frame = ui.to_frame()
    frame.iloc[:] = 999
    ui.model.solution.N[:] = 999
    np.testing.assert_allclose(ui.to_frame().x, .8 ** np.arange(4), atol=1e-13)
