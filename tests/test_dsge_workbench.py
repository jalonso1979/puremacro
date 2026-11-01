"""Independent experiment identity, replay, comparison and interruption checks."""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge import DSGEWorkbench, DSGERun, DSGERunControl, solution_accuracy
from puremacro.dsge.workbench import _digest

AR = """
var x; varexo e; parameters rho; rho=.8;
model; [name='Persistence'] x=rho*x(-1)+e; end;
initval; x=0; end;
"""
NONLINEAR = """
var x; varexo e;
model; x^2=.2*x(+1)+.3*x(-1)+.5+e; end;
initval; x=1; end;
"""


@pytest.mark.pyodide_smoke
def test_irf_history_is_frozen_and_matches_closed_form():
    bench = DSGEWorkbench.from_mod(AR)
    assert bench.state == "edited" and bench.last_attempt is None
    first = bench.irf("e", horizon=6, size=.01, label="Baseline")
    np.testing.assert_allclose(first.table().x, .01 * .8 ** np.arange(7), atol=1e-13)
    second = bench.irf("e", horizon=6, size=.01, params={"rho": .5}, label="Alternative")
    np.testing.assert_allclose(second.table().x, .01 * .5 ** np.arange(7), atol=1e-13)
    assert bench.runs == (first, second) and bench.last_success is second
    assert first.parameters == {"rho": .8}
    copy = first.table()
    copy.iloc[:] = 99
    first.to_dict()["inputs"]["parameters"]["rho"] = 99
    first.parameters["rho"] = 99
    np.testing.assert_allclose(first.table().x, .01 * .8 ** np.arange(7), atol=1e-13)
    bench.set_parameters({"rho": .7})
    assert bench.state == "edited" and bench.last_success is second
    assert not bench.history().current_parameters.any()


@pytest.mark.pyodide_smoke
def test_accuracy_does_not_turn_local_convergence_into_global_acceptance():
    run = DSGEWorkbench.from_mod(AR).irf("e", horizon=3)
    frame = run.accuracy.to_frame()
    assert (frame.loc[frame.dimension == "determinacy", "status"] == "passed").all()
    assert (frame.loc[frame.dimension == "equations", "status"] == "passed").all()
    for dim in ("horizon", "quadrature", "household_grid", "compression", "branching"):
        assert (frame.loc[frame.dimension == dim, "status"] == "not_checked").all()
    assert not hasattr(run.accuracy, "passed")
    unknown = solution_accuracy(SimpleNamespace(converged=True, diagnostics={"equation_converged": True}))
    assert "passed" not in set(unknown.to_frame().status)


def test_failed_attempt_has_no_stale_irf_and_preserves_previous_success():
    bench = DSGEWorkbench.from_mod(AR)
    good = bench.irf("e", horizon=3)
    bad = bench.irf("missing", horizon=3)
    assert bad.status == bench.state == "failed"
    assert bad.table_names == ()
    assert bad.failure.category == "model"
    assert "Blanchard-Kahn" not in bad.failure.message
    assert bench.last_attempt is bad and bench.last_success is good
    with pytest.raises(ValueError, match="successful"):
        good.compare(bad)


@pytest.mark.pyodide_smoke
def test_archive_is_data_only_and_explicit_replay_matches_closed_form(tmp_path, monkeypatch):
    run = DSGEWorkbench.from_mod(AR).irf("e", horizon=5, size=.02)
    path = run.save(tmp_path / "baseline.json")
    with pytest.raises(FileExistsError):
        run.save(path)
    with monkeypatch.context() as patch:
        patch.setattr(DSGEWorkbench, "from_mod", lambda *a, **kw: pytest.fail("load executed a model"))
        saved = DSGERun.load(path)
    assert saved.to_dict() == run.to_dict()
    replay = saved.replay()
    assert replay.run_id != saved.run_id and replay.status == "succeeded"
    np.testing.assert_allclose(replay.table().x, .02 * .8 ** np.arange(6), atol=1e-13)


def test_source_includes_are_frozen_for_replay(tmp_path):
    equations = tmp_path / "eq.mod"
    equations.write_text("model; x=rho*x(-1)+e; end;", encoding="utf-8")
    source = tmp_path / "model.mod"
    source.write_text('var x; varexo e; parameters rho; rho=.8;\n@#include "eq.mod"\ninitval; x=0; end;', encoding="utf-8")
    bench = DSGEWorkbench.from_mod(source)
    run = bench.irf("e", horizon=3)
    equations.write_text("invalid replacement", encoding="utf-8")
    source.unlink()
    replay = run.replay()
    np.testing.assert_allclose(replay.table().x, .8 ** np.arange(4), atol=1e-13)
    assert len(run.to_dict()["inputs"]["recipe"]["original_source_hashes"]) == 2


@pytest.mark.parametrize("tamper", ["table", "input", "schema"])
def test_archive_corruption_is_rejected(tmp_path, tamper):
    run = DSGEWorkbench.from_mod(AR).irf("e", horizon=2)
    path = run.save(tmp_path / "run.json")
    data = json.loads(path.read_text())
    if tamper == "table":
        data["payload"]["tables"]["irf"]["values"][0][0] += 1
    elif tamper == "input":
        data["payload"]["inputs"]["parameters"]["rho"] = .4
        data["sha256"] = _digest(data["payload"])
    else:
        data["schema"] = "unknown"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum|schema"):
        DSGERun.load(path)


def test_replay_requires_explicit_environment_change_and_portable_recipe():
    run = DSGEWorkbench.from_mod(AR).irf("e", horizon=2)
    data = run.to_dict()
    data["environment"]["numpy"] = "different"
    altered = replace(run, _json=json.dumps(data))
    with pytest.raises(ValueError, match="environment"):
        altered.replay()
    assert altered.replay(allow_environment_change=True).status == "succeeded"
    snapshot = DSGERun.capture(None, label="External result")
    with pytest.raises(ValueError, match="no portable"):
        snapshot.replay()


@pytest.mark.pyodide_smoke
def test_comparison_has_exact_analytic_differences_and_settings():
    bench = DSGEWorkbench.from_mod(AR)
    first = bench.irf("e", horizon=5, size=.01)
    second = bench.irf("e", horizon=5, size=.01, params={"rho": .5})
    comparison = first.compare(second)
    expected = .01 * (.5 ** np.arange(6) - .8 ** np.arange(6))
    np.testing.assert_allclose(comparison.paths()["difference"].x, expected, atol=1e-13)
    assert comparison.to_frame().loc["x", "max_abs_difference"] == pytest.approx(abs(expected).max())
    settings = comparison.settings()
    assert settings.loc["parameters.rho", "changed"]
    assert settings.loc["parameters.rho", "baseline"] == .8
    assert settings.loc["parameters.rho", "candidate"] == .5


@pytest.mark.parametrize("field,value", [("size", .5), ("shock", "other")])
def test_different_shock_experiments_require_explicit_comparison(field, value):
    first = DSGEWorkbench.from_mod(AR).irf("e", horizon=3)
    payload = first.to_dict()
    payload["inputs"]["configuration"][field] = value
    second = replace(first, _json=json.dumps(payload))
    with pytest.raises(ValueError, match="Experiment differs"):
        first.compare(second)
    assert first.compare(second, allow_experiment_change=True).settings().changed.any()


def test_comparison_rejects_unknown_units_and_requires_explicit_common_window():
    bench = DSGEWorkbench.from_mod(AR)
    a, b = bench.irf("e", horizon=3), bench.irf("e", horizon=5)
    with pytest.raises(ValueError, match="horizons"):
        a.compare(b)
    c = a.compare(b, alignment="inner")
    assert c.to_frame().attrs == dict(alignment="inner", compared_dates=4, baseline_dates=4, candidate_dates=6)
    payload = b.to_dict()
    payload["inputs"]["units"]["x"] = "percent"
    with pytest.raises(ValueError, match="units"):
        a.compare(replace(b, _json=json.dumps(payload)), alignment="inner")
    payload["inputs"]["timing"] = "unspecified"
    with pytest.raises(ValueError, match="timing"):
        a.compare(replace(b, _json=json.dumps(payload)), alignment="inner")


@pytest.mark.pyodide_smoke
def test_perfect_foresight_recipe_and_audit_match_independent_recursion():
    bench = DSGEWorkbench.from_mod(AR)
    eps = np.array([[.1], [.03], [0.], [-.02], [0.]])
    run = bench.perfect_foresight(periods=5, y_init=[0.], y_end=[0.], shocks=eps)
    expected, x = [], 0.
    for innovation in eps[:, 0]:
        x = .8 * x + innovation
        expected.append(x)
    assert run.status == "succeeded"
    np.testing.assert_allclose(run.table().x, expected, atol=1e-10)
    frame = run.accuracy.to_frame()
    assert "passed" in set(frame.loc[frame.dimension == "equations", "status"])
    assert "passed" not in set(frame.loc[frame.dimension == "horizon", "status"])
    pd.testing.assert_frame_equal(run.table().reset_index(drop=True), run.replay().table().reset_index(drop=True), check_flags=False)


def test_krylov_cancellation_retains_accepted_checkpoint_and_resume_rechecks():
    def cancel_after_step(event):
        if "phase" not in event:
            control.cancel()
    control = DSGERunControl(on_progress=cancel_after_step)
    bench = DSGEWorkbench.from_mod(NONLINEAR)
    options = dict(periods=6, y_init=[.9], y_end=[1.], shocks=np.full((6, 1), .1), linear_solver="krylov")
    stopped = bench.perfect_foresight(control=control, **options)
    assert stopped.status == "cancelled" and stopped.failure.category == "cancelled"
    checkpoint = stopped.to_dict()["execution"]["checkpoint"]
    assert np.asarray(checkpoint["path"]).shape == (6, 1)
    with pytest.raises(ValueError, match="one run"):
        bench.perfect_foresight(control=control, periods=3, y_init=[1.], y_end=[1.],
                                shocks=np.zeros((3, 1)), linear_solver="krylov")
    assert bench.runs == (stopped,)
    resumed = stopped.replay(resume=True)
    full = bench.perfect_foresight(**options)
    assert resumed.status == full.status == "succeeded"
    np.testing.assert_allclose(resumed.table(), full.table(), atol=1e-9)
    payload = stopped.to_dict()
    payload["execution"]["checkpoint"]["input_sha256"] = "different"
    with pytest.raises(ValueError, match="Checkpoint"):
        replace(stopped, _json=json.dumps(payload)).replay(resume=True)


def test_pre_cancel_and_cooperative_budget_have_distinct_states(monkeypatch):
    bench = DSGEWorkbench.from_mod(AR)
    control = DSGERunControl()
    control.cancel()
    assert bench.irf("e", control=control).status == "cancelled"
    import puremacro.dsge._workbench_state as state
    now = [0.]
    monkeypatch.setattr(state, "monotonic", lambda: now[0])
    control = DSGERunControl(max_seconds=1., on_progress=lambda event: now.__setitem__(0, 2.))
    run = bench.irf("e", control=control)
    assert run.status == "budget_exhausted" and run.failure.category == "budget"
    assert run.table_names == ()


def test_inputs_and_html_are_explicit_and_escaped():
    bench = DSGEWorkbench.from_mod(AR, name="<script>bad</script>")
    with pytest.raises(ValueError, match="Unknown"):
        bench.set_parameters({"invented": 1})
    with pytest.raises(ValueError, match="finite"):
        bench.set_parameters({"rho": np.nan})
    with pytest.raises(TypeError, match="portable"):
        bench.perfect_foresight(terminal_policy=lambda x: x)
    assert bench.parameters == {"rho": .8}
    run = bench.irf("e", horizon=2, label="<script>bad</script>")
    for html in (bench._repr_html_(), run._repr_html_(), run.accuracy._repr_html_(), run.to_html(), run.compare(run).to_html()):
        assert "<script>" not in html
    assert "&lt;script&gt;" in run._repr_html_()


@pytest.mark.pyodide_smoke
def test_exported_python_reproduces_declared_experiment():
    from puremacro.examples.dsge_workbench import run_example
    _, baseline, alternative = run_example()
    namespace = {}
    exec(compile(alternative.to_python(), "exported-experiment.py", "exec"), namespace)
    reproduced = namespace["run"]
    assert reproduced.parameters == alternative.parameters
    np.testing.assert_allclose(reproduced.table().output, .01 * .5 ** np.arange(21), atol=1e-13)
    assert "<!doctype html>" in baseline.to_html()
    assert "Showing 20 of 21" in baseline.to_html()
    with pytest.raises(ValueError, match="no portable"):
        DSGERun.capture(None).to_python()


@pytest.mark.parametrize("data", [[], {}, {"schema": "puremacro.dsge.run/1", "payload": {}}])
def test_malformed_archives_have_actionable_errors(tmp_path, data):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="schema|checksum|payload"):
        DSGERun.load(path)


def test_unknown_external_results_are_recorded_without_inventing_success():
    snapshot = DSGERun.capture(SimpleNamespace(converged=True))
    assert snapshot.status == "recorded" and not snapshot.to_dict()["replayable"]


def test_global_policy_dashboard_preserves_pointwise_evidence():
    from puremacro.dsge import EquilibriumProblem, solve_time_iteration
    problem = EquilibriumProblem(("state",), ("control",), [[0., 1.]],
        lambda s, x, e: s, lambda s, x, sp, xp, e: x - (.2 + .5 * s))
    policy = solve_time_iteration(problem, lambda s: np.array([.4]), orders=3, tol=1e-9)
    np.testing.assert_allclose(policy([[.1], [.6]]), [[.25], [.5]], atol=1e-9)
    frame = solution_accuracy(policy).to_frame().set_index("check")
    assert frame.loc["policy_kkt_residual", "status"] == "passed"
    assert frame.loc["quadrature", "status"] == "not_checked"


def test_transition_dashboard_audits_the_solved_path():
    from puremacro.dsge import read_mod, solve_transition
    result = solve_transition(read_mod(AR), y_init=[0.], y_end=[0.], shocks=np.ones((4, 1)))
    assert result.converged
    expected = np.cumsum(.8 ** np.arange(4))
    np.testing.assert_allclose(result.path.x, expected, atol=1e-9)
    frame = solution_accuracy(result).to_frame()
    assert "passed" in set(frame.loc[frame.dimension == "equations", "status"])
    assert "passed" not in set(frame.loc[frame.dimension == "horizon", "status"])


def test_hank_dashboard_separates_finite_equations_and_household_accuracy():
    from puremacro.models import calibrate_one_asset_hank, solve_hank_stochastic_policy, solve_hank_nonlinear_continuation
    steady = calibrate_one_asset_hank(n_a=20, n_e=2, a_max=120.)
    policy = solve_hank_stochastic_policy(steady, distribution_rank=None, shock_std=1e-5)
    correction = solve_hank_nonlinear_continuation(policy, distribution=steady.distribution,
        aggregate_state=np.zeros(2), horizon=3, branch_depth=1)
    assert correction.converged
    dashboard = solution_accuracy(correction)
    frame = dashboard.to_frame()
    euler = frame.loc[frame['check'] == 'euler_kkt_max'].iloc[0]
    assert euler.status == "reported" and euler.value > 0
    assert "passed" not in set(frame.loc[frame.dimension.isin(["horizon", "branching", "quadrature"]), "status"])
    strict = solution_accuracy(correction, tolerances={"euler": 0.}).to_frame()
    assert strict.loc[strict['check'] == 'euler_kkt_max', 'status'].item() == "failed"
