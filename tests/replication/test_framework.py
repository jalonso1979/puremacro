"""Validates the replication framework itself (the replicator)."""
import numpy as np

from puremacro.replication._model import (
    CaseResult,
    ReplicationCase,
    TargetKind,
    Tol,
    _compare_target,
    run,
)


# --- Task 1: target kinds + comparator ---


def test_sign_kind():
    assert _compare_target({"d": 0.012}, {"d": +1}, TargetKind.SIGN, Tol.COARSE)[0] is True
    assert _compare_target({"d": -0.012}, {"d": +1}, TargetKind.SIGN, Tol.COARSE)[0] is False


def test_correlation_kind_is_lower_bound():
    assert _compare_target({"c": 0.95}, {"c": 0.90}, TargetKind.CORRELATION, Tol.COARSE)[0] is True
    assert _compare_target({"c": 0.80}, {"c": 0.90}, TargetKind.CORRELATION, Tol.COARSE)[0] is False


def test_point_kind_respects_rtol():
    # 7.5% off: passes COARSE (25%), fails TIGHT (2%)
    assert _compare_target({"r": 0.043}, {"r": 0.040}, TargetKind.POINT, Tol.COARSE)[0] is True
    assert _compare_target({"r": 0.043}, {"r": 0.040}, TargetKind.POINT, Tol.TIGHT)[0] is False


def test_compare_only_checks_target_keys():
    assert _compare_target({"r": 0.04, "extra": 9.9}, {"r": 0.04}, TargetKind.POINT, Tol.COARSE)[0] is True


def test_target_kind_values():
    assert {k.value for k in TargetKind} == {"point", "sign", "correlation"}


# --- Task 2: ReplicationCase + run ---


def _case(estimate, target, kind=TargetKind.SIGN, tol=Tol.COARSE):
    return ReplicationCase(
        id="t.case", family="t", paper="Test (2026)", title="t", title_es="t",
        source="model", estimate=estimate, target=target, target_kind=kind, tol=tol,
        citation="unit-test",
    )


def test_run_passing_case():
    r = run(_case(lambda: {"d": 0.02}, {"d": +1}))
    assert isinstance(r, CaseResult) and r.passed is True and r.error == ""


def test_run_failing_case():
    r = run(_case(lambda: {"d": -0.02}, {"d": +1}))
    assert r.passed is False and r.error == ""


def test_run_captures_estimate_exception():
    def boom():
        raise RuntimeError("solve diverged")

    r = run(_case(boom, {"d": +1}))
    assert r.passed is False and "solve diverged" in r.error


# --- Task 3: runner (run_all + scorecard) ---


from puremacro.replication import runner as _runner  # noqa: E402


def test_run_all_discovers_and_filters(monkeypatch):
    demo = ReplicationCase(
        id="demo.ok", family="demo", paper="P (2026)", title="d", title_es="d",
        source="model", estimate=lambda: {"d": 0.02}, target={"d": +1},
        target_kind=TargetKind.SIGN, tol=Tol.COARSE, citation="ut",
    )
    monkeypatch.setattr(_runner, "_discover_cases", lambda: [demo])
    res = _runner.run_all()
    assert len(res) == 1 and res[0].passed
    assert _runner.run_all(family="nope") == []


def test_scorecard_columns(monkeypatch):
    demo = ReplicationCase(
        id="demo.ok", family="demo", paper="P (2026)", title="d", title_es="d",
        source="model", estimate=lambda: {"d": 0.02}, target={"d": +1},
        target_kind=TargetKind.SIGN, tol=Tol.COARSE, citation="ut",
    )
    monkeypatch.setattr(_runner, "_discover_cases", lambda: [demo])
    df = _runner.scorecard()
    assert list(df.columns) == [
        "id", "family", "paper", "source", "target_kind", "tol", "passed", "margin", "network", "citation"
    ]
    assert bool(df.loc[0, "passed"]) is True


# --- Task 4: public surface + purity ---


def test_public_surface_importable():
    import puremacro.replication as R

    for name in ("run", "run_all", "scorecard", "ReplicationCase", "CaseResult", "TargetKind", "Tol"):
        assert hasattr(R, name), name


def test_replication_imports_no_heavy_deps():
    import importlib
    import sys

    for m in list(sys.modules):
        if m.startswith(("statsmodels", "linearmodels", "arch")):
            del sys.modules[m]
    importlib.import_module("puremacro.replication")
    importlib.import_module("puremacro.replication.runner")
    leaked = [m for m in sys.modules if m.startswith(("statsmodels", "linearmodels", "arch"))]
    assert leaked == [], f"replication leaked heavy deps: {leaked}"


# --- Task 5: pyproject marker ---


def test_replication_marker_registered():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]  # .../puremacro (package dir)
    cfg = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    ini = cfg["tool"]["pytest"]["ini_options"]
    assert any(m.startswith("replication:") for m in ini["markers"])
    assert "not replication" in " ".join(ini["addopts"])
