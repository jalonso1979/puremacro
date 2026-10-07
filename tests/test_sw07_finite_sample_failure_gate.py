"""Successful optimizer termination is insufficient when stationarity fails."""
from types import SimpleNamespace

import numpy as np

from puremacro.structural.bridge import MomentTargets
from puremacro.structural.empirical_sw07 import ALL_MOMENTS, _labels
import puremacro.structural.sw07_finite_sample as module


def _targets():
    return MomentTargets(np.zeros(15), np.eye(15), _labels(ALL_MOMENTS), "test covariance units")


def _result(*, objective, reasons, boundary=False):
    return SimpleNamespace(theta=np.array([.98 if boundary else .7, .2]),
                           objective=objective, success=True, inference_valid=False,
                           message="SciPy reports successful termination", n_evals=8,
                           diagnostics={"inference_unavailable_reasons": tuple(reasons)},
                           boundary=np.array([boundary, False]), identification_rank=2,
                           j_pvalue=np.nan)


def test_selected_successful_but_nonstationary_solution_is_unresolved(monkeypatch):
    calls = []

    def engine(*args, **kwargs):
        first = not calls
        calls.append(args[2])
        return _result(objective=1. if first else 1.+1e-9,
                       reasons=("nonstationary_solution",) if first else ())

    monkeypatch.setattr(module, "fit_structural", engine)
    result, diagnostics = module._fitter({})(_targets())
    record = module._fit_record(result, diagnostics)
    assert len(calls) == 4 and diagnostics["start_successes"] == 4
    assert diagnostics["starts_agree"]
    assert diagnostics["unresolved"]
    assert record["status"] == "optimization_unresolved"
    assert not record["numerically_regular"]
    assert "stationarity" in record["error"]
    assert diagnostics["start_records"][0]["inference_unavailable_reasons"] == ["nonstationary_solution"]


def test_successful_boundary_solution_remains_usable_criterion(monkeypatch):
    monkeypatch.setattr(module, "fit_structural", lambda *args, **kwargs:
                        _result(objective=1., reasons=("parameter_on_boundary",), boundary=True))
    result, diagnostics = module._fitter({})(_targets())
    record = module._fit_record(result, diagnostics)
    assert diagnostics["start_successes"] == 4 and diagnostics["starts_agree"]
    assert not diagnostics["unresolved"]
    assert record["status"] == "boundary" and record["boundary"]
    assert not record["numerically_regular"] and np.isnan(record["j_pvalue"])
    assert np.isfinite(record["objective"])
