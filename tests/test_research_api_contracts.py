"""Cross-module API boundaries and retained research-result integrity."""
from dataclasses import replace
import importlib
import json

import numpy as np
import pytest

from puremacro.core import EquilibriumSystem, linearize_equilibrium, continue_equilibrium
from puremacro.expectations import GaussianSignal, build_gaussian_information, audit_gaussian_information
from puremacro.models import InformationNKParameters, solve_information_nk, audit_information_nk
from puremacro.sovereign import (
    SovereignBankingParameters, SovereignAdaptationParameters,
    solve_sovereign_banking, solve_sovereign_adaptation,
)
from puremacro.structural import MomentTargets, SimulationPlan, simulate_moments
from puremacro.trade import (
    ReallocationEconomy, ReallocationHorizonResult, audit_reallocation,
    solve_reallocation_steady_state, solve_reallocation_transition,
)


@pytest.fixture(scope="module")
def retained():
    system = EquilibriumSystem(lambda x, p: x - p, ("x",), ("p",))
    linear = linearize_equilibrium(system, [1.], [1.])
    info = build_gaussian_information(np.eye(2), [[GaussianSignal(0, [1., 0.], "news")]], horizon=3)
    nk = solve_information_nk(info, [[.01, 0.], [0., 0.], [0., 0.]])
    economy = ReallocationEconomy()
    steady = solve_reallocation_steady_state(economy)
    trade = solve_reallocation_transition(economy, np.zeros((3, 2, 2)), initial=steady, terminal=steady)
    banking = solve_sovereign_banking(income=[.8, 1.], productivity=[.4, .6],
        probabilities=[.4, .6], debt_grid=[0., .1, .2], price_selection="highest")
    adaptation = solve_sovereign_adaptation(income=[1., 1.], damage=[0., .2], disaster=[False, True],
        transition=[[.9, .1], [.3, .7]], debt_grid=[0., .1], adaptation_grid=[0., .05], horizon=3)
    targets = MomentTargets([1.], [[.1]], ("mean",))
    batch = simulate_moments(lambda x, rng, plan: x + rng.normal(size=1), [1.],
        SimulationPlan(seed=9, replications=4, periods=1), labels=targets.labels, definition_id="location")
    return dict(linear=linear, branch=continue_equilibrium(system, [1.], [1.], max_steps=2),
        signal=info.signals[0][0], info=info, nk=nk, nk_path=nk.evaluate([1., 0.]),
        economy=economy, steady=steady, trade=trade, banking=banking,
        bank_equilibrium=banking.selected, adaptation=adaptation,
        distribution=adaptation.distribution([1., 0.]), targets=targets, batch=batch)


@pytest.mark.parametrize("owner,field", [
    ("linear", "state"), ("branch", "states"), ("signal", "loading"),
    ("info", "covariance"), ("info", "projectors"), ("info", "ranks"),
    ("nk", "output_loadings"), ("nk_path", "output"), ("economy", "productivity"),
    ("steady", "coordinates"), ("trade", "coordinates"), ("banking", "income"),
    ("banking", "selected_indices"), ("bank_equilibrium", "default"),
    ("bank_equilibrium", "accounts"), ("adaptation", "prices"),
    ("adaptation", "default"), ("adaptation", "repayment_debt_policy"),
    ("distribution", "access_mass"), ("targets", "values"), ("targets", "covariance"),
    ("batch", "replicates"),
])
def test_retained_arrays_cannot_be_made_writable(retained, owner, field):
    array = getattr(retained[owner], field)
    with pytest.raises(ValueError):
        array.setflags(write=True)
    # Protection must survive following ndarray views down to their buffer.
    base = array
    while isinstance(base.base, np.ndarray):
        base = base.base
        with pytest.raises(ValueError):
            base.setflags(write=True)


@pytest.mark.parametrize("owner", ["info", "nk", "steady", "trade", "banking", "adaptation", "distribution"])
def test_nested_diagnostics_are_detached_and_readonly(retained, owner):
    supplied = {"checks": {"gap": [1., 2.]}, "array": np.array([3.])}
    result = replace(retained[owner], diagnostics=supplied)
    supplied["checks"]["gap"][0] = 99.
    supplied["array"][0] = 99.
    assert result.diagnostics["checks"]["gap"][0] == 1.
    assert result.diagnostics["array"][0] == 3.
    with pytest.raises(TypeError):
        result.diagnostics["checks"]["gap"][0] = 5.
    with pytest.raises(ValueError):
        result.diagnostics["array"].setflags(write=True)


def test_horizon_attempt_table_does_not_alias_retained_failures():
    attempt = {"solved": False, "diagnostics": {"checks": {"gap": 1.}}}
    result = ReallocationHorizonResult((), (attempt,), "unresolved")
    attempt["diagnostics"]["checks"]["gap"] = 10.
    assert result.attempts[0]["diagnostics"]["checks"]["gap"] == 1.
    frame = result.to_frame()
    frame.iloc[0]["diagnostics"]["checks"]["gap"] = 20.
    assert result.attempts[0]["diagnostics"]["checks"]["gap"] == 1.


@pytest.mark.parametrize("bad", [True, np.bool_(True), "0.5", np.array("0.5"), [0.5],
                                      np.array([.5]), 1+0j, None, float("nan"), float("inf"), 10**1000],
                         ids=["bool", "numpy_bool", "string", "string_scalar", "list", "array", "complex",
                              "none", "nan", "infinity", "overflow"])
@pytest.mark.parametrize("consumer", ["linearization", "information_nk", "trade", "banking", "adaptation"])
def test_real_scalar_controls_reject_non_scalars_and_coercions(bad, consumer):
    with pytest.raises(ValueError):
        if consumer == "linearization":
            linearize_equilibrium(EquilibriumSystem(lambda x, p: x-p, ("x",), ("p",)), [1.], [1.], step=bad)
        else:
            factory = {"information_nk": InformationNKParameters, "trade": ReallocationEconomy,
                       "banking": SovereignBankingParameters, "adaptation": SovereignAdaptationParameters}[consumer]
            factory(beta=bad)


def test_equilibrium_labels_and_switches_are_explicit(retained):
    with pytest.raises(ValueError):
        EquilibriumSystem(lambda x, p: x-p, "capital", ("p",))
    with pytest.raises(ValueError):
        retained["linear"].solve([1.], transpose="False")
    with pytest.raises(ValueError):
        continue_equilibrium(retained["linear"].system, [1.], [1.], direction=np.bool_(True), max_steps=1)


@pytest.mark.parametrize("change", [
    {"covariance": np.array(1.)}, {"weights": np.array(1.)}, {"horizon": True},
    {"group_names": ()}, {"state_names": ("duplicate", "duplicate")},
])
def test_information_audit_rejects_malformed_retained_metadata(retained, change):
    report = audit_gaussian_information(replace(retained["info"], **change))
    assert report["accepted"] is False
    assert report["reason"]


def test_nk_audit_rejects_invalid_retained_parameters(retained):
    report = audit_information_nk(replace(retained["nk"], parameters=None))
    assert report["accepted"] is False


@pytest.mark.parametrize("change", ["missing_flow", "flow_shape", "coordinates", "tariffs", "initial", "terminal_model"])
def test_trade_audit_rejects_malformed_retained_results(retained, change):
    run = retained["trade"]
    if change in ("missing_flow", "flow_shape"):
        paths = dict(run.paths)
        if change == "missing_flow":
            paths.pop("consumption")
        else:
            paths["consumption"] = paths["consumption"][:, :1]
        run = replace(run, paths=paths)
    elif change == "coordinates":
        run = replace(run, coordinates=run.coordinates[:, :-1])
    elif change == "tariffs":
        run = replace(run, tariffs=np.zeros((3, 2)))
    elif change == "initial":
        run = replace(run, initial=None)
    else:
        run = replace(run, terminal=replace(run.terminal, economy=replace(run.economy, beta=.95)))
    report = audit_reallocation(run)
    assert report["accepted"] is False
    assert report["reason"]


@pytest.mark.parametrize("name", ["simulation", "robust"])
def test_statistical_modules_declare_only_their_public_exports(name):
    module = importlib.import_module("puremacro.structural." + name)
    parent = importlib.import_module("puremacro.structural")
    assert module.__all__
    assert len(module.__all__) == len(set(module.__all__))
    for symbol in module.__all__:
        assert getattr(module, symbol) is getattr(parent, symbol)
        assert symbol in parent.__all__
    assert not {"np", "pd", "Callable", "chi2", "norm"}.intersection(module.__all__)


def test_example_json_serializers_preserve_nested_diagnostics(retained, tmp_path):
    from puremacro.examples.heterogeneous_expectations import _write
    from puremacro.examples.trade_reallocation import _write_json
    from puremacro.examples.sovereign_banking import save_result, load_result
    from puremacro.examples.sovereign_adaptation import save_result as save_adaptation, load_result as load_adaptation

    _write(tmp_path/"information.json", dict(retained["info"].diagnostics))
    _write_json(tmp_path/"trade.json", dict(retained["trade"].diagnostics))
    assert json.loads((tmp_path/"information.json").read_text())["accepted"]
    assert json.loads((tmp_path/"trade.json").read_text())["accepted"]
    save_result(retained["banking"], tmp_path/"banking")
    assert load_result(tmp_path/"banking").fingerprint == retained["banking"].fingerprint
    save_adaptation(retained["adaptation"], retained["distribution"], tmp_path/"adaptation")
    restored, distribution = load_adaptation(tmp_path/"adaptation")
    assert restored.fingerprint == retained["adaptation"].fingerprint
    assert distribution.value == retained["distribution"].value
