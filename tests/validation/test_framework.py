"""Validates the validation framework itself (the validator)."""
import numpy as np
import pytest

from puremacro.validation._goldens import load_golden
from puremacro.validation._model import (
    CaseResult,
    Mechanism,
    Tol,
    ValidationCase,
    _compare,
    run,
)


# --- Task 1: comparison core ---


def test_compare_exact_pass_and_fail():
    assert _compare({"x": 1.0}, {"x": 1.0}, Tol.EXACT)[0] is True
    ok, margin = _compare({"x": 1.0}, {"x": 1.0 + 1e-6}, Tol.EXACT)
    assert ok is False and margin > 0


def test_compare_tight_tolerates_tiny_drift():
    ok, _ = _compare({"x": 1.0}, {"x": 1.0 + 1e-8}, Tol.TIGHT)
    assert ok is True


def test_compare_arrays_elementwise():
    a = {"irf": np.zeros((3, 2))}
    b = {"irf": [[0.0, 0.0], [0.0, 0.0], [0.0, 1e-3]]}
    assert _compare(a, b, Tol.TIGHT)[0] is False
    assert _compare(a, b, Tol.COARSE)[0] is False  # 1e-3 vs 0 fails even at 10% (atol 0)


def test_compare_qualitative_is_lower_bound():
    assert _compare({"corr": 0.9}, {"corr": 0.8}, Tol.QUALITATIVE)[0] is True
    assert _compare({"corr": 0.7}, {"corr": 0.8}, Tol.QUALITATIVE)[0] is False


def test_compare_only_checks_reference_keys():
    # pm may carry extra keys; only ref keys are asserted
    assert _compare({"x": 1.0, "extra": 999.0}, {"x": 1.0}, Tol.EXACT)[0] is True


def test_mechanism_values():
    assert {m.value for m in Mechanism} == {
        "package",
        "scipy",
        "analytical",
        "published",
        "internal",
    }


# --- Task 2: ValidationCase + run ---


def _case(compute, reference, tol=Tol.EXACT):
    return ValidationCase(
        id="t.case",
        subsystem="t",
        title="t",
        title_es="t",
        mechanism=Mechanism.INTERNAL,
        compute=compute,
        reference=reference,
        tol=tol,
        citation="unit-test",
    )


def test_run_passing_case():
    r = run(_case(lambda: {"x": 1.0}, lambda: {"x": 1.0}))
    assert isinstance(r, CaseResult)
    assert r.passed is True and r.error == "" and r.id == "t.case"


def test_run_failing_case():
    r = run(_case(lambda: {"x": 1.0}, lambda: {"x": 2.0}))
    assert r.passed is False and r.error == ""


def test_run_captures_compute_exception():
    def boom():
        raise ValueError("kaboom")

    r = run(_case(boom, lambda: {"x": 1.0}))
    assert r.passed is False and "kaboom" in r.error


# --- Task 3: goldens loader ---


def test_load_golden_returns_metrics():
    d = load_golden("_t:demo")
    assert d["x"] == 1.5 and d["vec"] == [1.0, 2.0, 3.0]


def test_load_golden_missing_case_raises():
    with pytest.raises(KeyError):
        load_golden("_t:nope")


# --- Task 4: runner (run_all + scorecard) ---


from puremacro.validation import runner as _runner  # noqa: E402


def test_run_all_discovers_cases_and_filters(monkeypatch):
    demo = ValidationCase(
        id="demo.ok", subsystem="demo", title="d", title_es="d",
        mechanism=Mechanism.INTERNAL, compute=lambda: {"x": 1.0},
        reference=lambda: {"x": 1.0}, tol=Tol.EXACT, citation="ut",
    )
    monkeypatch.setattr(_runner, "_discover_cases", lambda: [demo])
    res = _runner.run_all()
    assert len(res) == 1 and res[0].passed
    assert _runner.run_all(subsystem="nope") == []


def test_scorecard_columns(monkeypatch):
    demo = ValidationCase(
        id="demo.ok", subsystem="demo", title="d", title_es="d",
        mechanism=Mechanism.PACKAGE, compute=lambda: {"x": 1.0},
        reference=lambda: {"x": 1.0}, tol=Tol.TIGHT, citation="ut",
    )
    monkeypatch.setattr(_runner, "_discover_cases", lambda: [demo])
    df = _runner.scorecard()
    assert list(df.columns) == [
        "id", "subsystem", "mechanism", "tol", "passed", "max_margin", "citation"
    ]
    assert bool(df.loc[0, "passed"]) is True


# --- Task 5: public surface + purity ---


def test_public_surface_importable():
    import puremacro.validation as V

    for name in ("run_all", "scorecard", "ValidationCase", "CaseResult", "Mechanism", "Tol"):
        assert hasattr(V, name), name


def test_validation_imports_no_heavy_deps():
    # Importing the validation package must not pull statsmodels/linearmodels/arch.
    import importlib
    import sys

    for m in list(sys.modules):
        if m.startswith(("statsmodels", "linearmodels", "arch")):
            del sys.modules[m]
    importlib.import_module("puremacro.validation")
    importlib.import_module("puremacro.validation.runner")
    leaked = [m for m in sys.modules if m.startswith(("statsmodels", "linearmodels", "arch"))]
    assert leaked == [], f"validation leaked heavy deps: {leaked}"


# --- Task 6: pyproject marker + package-data ---


def test_reference_marker_registered():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]  # .../puremacro (package dir)
    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    ini = cfg["tool"]["pytest"]["ini_options"]
    assert any(m.startswith("reference:") for m in ini["markers"])
    assert "not reference" in " ".join(ini["addopts"])
    pkgdata = cfg["tool"]["setuptools"]["package-data"]
    assert pkgdata.get("puremacro.validation.goldens") == ["*.json"]


# --- Task 8: shared var DGP fixture ---


def test_var_demo_data_deterministic():
    from puremacro.validation._fixtures import var_demo_data

    a = var_demo_data()
    b = var_demo_data()
    assert a["Y"].shape == (300, 2)
    assert a["p"] == 2 and a["horizon"] == 12
    np.testing.assert_array_equal(a["Y"], b["Y"])  # seeded → identical


# --- Task 9: var golden generated + provenance ---


def test_var_golden_has_cholesky_irf_and_provenance():
    d = load_golden("var:cholesky_irf_vs_statsmodels")
    irf = np.asarray(d["irf"], dtype=float)
    assert irf.shape == (13, 2, 2)  # (horizon+1, k, k)
    import json
    from importlib import resources

    raw = json.loads(
        resources.files("puremacro.validation.goldens").joinpath("var.json").read_text(encoding="utf-8")
    )
    assert "statsmodels" in raw["_meta"]["reference"]
