"""Tests for identification-robust structural inference, SMM, HTML, and bundle export in workbench."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from puremacro.structural.bridge import MomentTargets
from puremacro.structural.robust import QuadraticMomentCurve
from puremacro.structural.simulation import SimulatedMoments, SimulationPlan
from puremacro.workbench import (
    EmpiricalStructuralDossier,
    run_empirical_structural_workbench,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_ts_df(n: int = 80, *, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq="QS")
    x = np.zeros(n)
    y = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.4 * x[t - 1] + rng.normal(0.0, 0.8)
        y[t] = 0.35 * y[t - 1] - 0.7 * x[t] - 0.25 * x[t - 1] + rng.normal(0.0, 0.3)
    return pd.DataFrame({"y": y, "x": x}, index=idx)


def _simple_targets() -> MomentTargets:
    return MomentTargets(
        values=np.array([1.0, 0.0]),
        covariance=np.array([[0.25, 0.0], [0.0, 0.25]]),
        labels=("m1", "m2"),
        units=("pct", "pct"),
    )


def test_workbench_backward_compatible_and_lazy_imports() -> None:
    df = _make_ts_df()
    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(5),
        lags=1,
        run_sensitivity=False,
    )
    assert isinstance(dossier, EmpiricalStructuralDossier)
    assert dossier.moment_test is None
    assert dossier.moment_confidence_set is None
    assert dossier.moment_grid is None
    assert dossier.smm_fit is None
    d = dossier.to_dict()
    for key in ("moment_test", "moment_confidence_set", "moment_grid", "smm_fit"):
        assert d[key] is None
    json.dumps(d, allow_nan=False)

    # Fresh subprocess importing puremacro.workbench keeps structural.robust & simulation unloaded
    probe = (
        "import json, sys; "
        "import puremacro.workbench; "
        "print(json.dumps({'robust': 'puremacro.structural.robust' in sys.modules, "
        "'sim': 'puremacro.structural.simulation' in sys.modules, "
        "'structural': 'puremacro.structural' in sys.modules}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    res = json.loads(proc.stdout)
    assert res == {"robust": False, "sim": False, "structural": False}


def test_workbench_moment_test_and_invalid_simulation_preserved() -> None:
    df = _make_ts_df()
    targets = _simple_targets()

    dossier_ok = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(4),
        lags=1,
        run_sensitivity=False,
        moment_test_targets=targets,
        model_moments=[1.1, 0.05],
    )
    assert dossier_ok.moment_test is not None
    assert dossier_ok.moment_test.accepted is True
    assert dossier_ok.moment_test.status == "valid"
    mt_df = dossier_ok.to_frame("moment_test")
    assert len(mt_df) == 1
    assert bool(mt_df["accepted"].iloc[0]) is True

    # Rejection produces explicit warning without falsifying passed preflight
    dossier_rej = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(4),
        lags=1,
        run_sensitivity=False,
        moment_test_targets=targets,
        model_moments=[8.0, 0.0],
    )
    assert dossier_rej.passed is True
    assert dossier_rej.overall_status == "PASS_WITH_WARNINGS"
    assert dossier_rej.moment_test is not None
    assert dossier_rej.moment_test.accepted is False
    assert any("rejected null" in w for w in dossier_rej.warnings)

    # Simulation with failed replicate status preserves evidence with accepted=None
    plan = SimulationPlan(seed=7, replications=4, periods=20)
    sim_bad = SimulatedMoments(
        theta=np.array([1.0]),
        plan=plan,
        labels=("m1", "m2"),
        units=("pct", "pct"),
        definition_id="sim_test",
        replicates=np.array([[1.0, 0.0], [1.1, 0.1], [0.9, -0.1], [1.0, 0.0]]),
        counts=np.full((4, 2), np.nan),
        statuses=("ok", "solver_failed", "ok", "ok"),
    )
    dossier_sim = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(4),
        lags=1,
        run_sensitivity=False,
        moment_test_targets=targets,
        simulation=sim_bad,
    )
    assert dossier_sim.moment_test is not None
    assert dossier_sim.moment_test.status == "invalid_simulation"
    assert dossier_sim.moment_test.accepted is None
    assert dossier_sim.moment_test.simulation is sim_bad
    sim_json = dossier_sim.to_dict()["moment_test"]
    assert sim_json["accepted"] is None
    assert sim_json["simulation"]["valid"] is False
    assert sim_json["simulation"]["statuses"][1] == "solver_failed"
    json.dumps(dossier_sim.to_dict(), allow_nan=False)


def test_workbench_quadratic_confidence_set_disconnected_truncated_and_singular() -> None:
    df = _make_ts_df()
    # Two disconnected intervals: m1(theta) = theta^2, target m1=4, m2(theta)=0
    targets = MomentTargets(
        values=np.array([4.0]),
        covariance=np.array([[1.0]]),
        labels=("m1",),
    )
    curve = QuadraticMomentCurve(
        offset=np.array([0.0]),
        loading=np.array([1.0]),
        labels=("m1",),
        definition_id="quad_disconnected",
        linear=0.0,
        quadratic=1.0,
        parameter_name="phi",
    )

    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(4),
        lags=1,
        run_sensitivity=False,
        moment_test_targets=targets,
        moment_curve=curve,
        moment_domain=(-2.2, 10.0),
    )
    cs = dossier.moment_confidence_set
    assert cs is not None
    assert cs.resolved is True
    assert len(cs.intervals) == 2
    assert cs.truncated is True
    assert cs.boundary_contact == (True, False)
    assert "truncated" in " ".join(dossier.warnings)

    cs_df = dossier.to_frame("confidence_set")
    assert len(cs_df) == 2
    assert set(cs_df["lower_source"]) == {"search_boundary", "test_boundary"}

    # Singular covariance -> unresolved status preserved, no pseudoinverse masking
    targets_sing = MomentTargets(
        values=np.array([1.0, 2.0]),
        covariance=np.array([[1.0, 1.0], [1.0, 1.0]]),
        labels=("a", "b"),
    )
    curve_sing = QuadraticMomentCurve(
        offset=np.array([0.0, 0.0]),
        loading=np.array([1.0, 1.0]),
        labels=("a", "b"),
        definition_id="sing_test",
        parameter_name="gamma",
    )
    dossier_sing = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(4),
        lags=1,
        run_sensitivity=False,
        moment_test_targets=targets_sing,
        moment_curve=curve_sing,
    )
    assert dossier_sing.moment_confidence_set is not None
    assert dossier_sing.moment_confidence_set.resolved is False
    assert dossier_sing.moment_confidence_set.status == "unresolved_covariance"
    assert "gamma" in dossier_sing.to_frame("confidence_set")["parameter"].tolist()
    json.dumps(dossier_sing.to_dict(), allow_nan=False)


def test_workbench_moment_grid_retains_unresolved_cells() -> None:
    from puremacro.structural.robust import moment_restriction_test

    df = _make_ts_df()
    targets = MomentTargets(
        values=np.array([4.0]),
        covariance=np.array([[0.25]]),
        labels=("m1",),
    )
    grid_vals = np.array([-2.1, -1.9, 0.0, 2.0, 5.0])

    def _eval_node(theta: float):
        if theta > 4.0:
            return moment_restriction_test(targets, [float("nan")])
        return moment_restriction_test(targets, [theta ** 2])

    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(4),
        lags=1,
        run_sensitivity=False,
        moment_test_targets=targets,
        moment_grid_fn=_eval_node,
        moment_grid_values=grid_vals,
        moment_parameter_name="beta",
    )
    mg = dossier.moment_grid
    assert mg is not None
    assert mg.continuous_complete is False
    assert dossier.moment_confidence_set is None
    assert len(dossier.to_frame("moment_grid")) == 5
    cells = dossier.to_frame("moment_grid_cells")
    assert len(cells) == 4
    assert list(cells["reason"]) == [
        "unsampled_interior",
        "unsampled_interior",
        "unsampled_interior",
        "invalid_endpoint",
    ]
    assert any("continuous_complete=False" in w for w in dossier.warnings)
    assert "beta" in dossier.summary(lang="en")
    assert "beta" in dossier.summary(lang="es")


def test_workbench_smm_integration() -> None:
    df = _make_ts_df()
    targets = MomentTargets(
        values=np.array([2.0, 3.0]),
        covariance=np.diag([0.05, 0.08]),
        labels=("m1", "m2"),
    )

    def simulator(theta: np.ndarray, rng: np.random.Generator, plan: SimulationPlan) -> np.ndarray:
        del plan
        rng_noise = float(rng.normal(0.0, 1e-3))
        m1 = float(theta[0]) + rng_noise
        m2 = 1.5 * float(theta[0]) + rng_noise
        return np.array([m1, m2])

    plan = SimulationPlan(seed=42, replications=8, periods=20)
    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(4),
        lags=1,
        run_sensitivity=False,
        moment_test_targets=targets,
        smm_simulator=simulator,
        smm_theta0=np.array([1.2]),
        smm_plan=plan,
        smm_definition_id="smm_lin_test",
        smm_kwargs={
            "parameter_names": ("alpha",),
            "assume_independent": True,
            "assume_smooth": True,
        },
    )
    assert dossier.smm_fit is not None
    assert dossier.smm_fit.success is True
    assert dossier.smm_fit.inference_valid is True
    assert np.isclose(dossier.smm_fit.theta[0], 2.0, atol=1e-2)
    fit_df = dossier.to_frame("smm_fit")
    assert list(fit_df["parameter"]) == ["alpha"]
    mom_df = dossier.to_frame("smm_moments")
    assert list(mom_df["label"]) == ["m1", "m2"]
    d = dossier.to_dict()
    assert d["smm_fit"]["success"] is True
    assert d["smm_fit"]["inference_valid"] is True
    json.dumps(d, allow_nan=False)


def test_workbench_html_escaping_and_bilingual_bundle_zero_host_paths(tmp_path: Path) -> None:
    df = _make_ts_df()
    targets = _simple_targets()
    curve = QuadraticMomentCurve(
        offset=np.array([0.0, 0.0]),
        loading=np.array([1.0, 0.5]),
        labels=("m1", "m2"),
        definition_id="bundle_curve",
        parameter_name="psi",
        units=("pct", "pct"),
    )
    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=range(4),
        lags=1,
        run_sensitivity=False,
        moment_test_targets=targets,
        model_moments=[1.0, 0.0],
        moment_curve=curve,
        moment_domain=(-2.0, 4.0),
        title='Robust <script>alert("xss")</script> & Español',
    )

    html_en = dossier.to_html(lang="en")
    html_es = dossier.to_html(lang="es")
    assert "<script>" not in html_en
    assert "&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;" in html_en
    assert 'lang="en"' in html_en
    assert 'lang="es"' in html_es
    assert "Conjunto de confianza robusto" in html_es
    assert "Identification-Robust Confidence Set" in html_en

    bundle_dir = tmp_path / "secret_host_dir" / "export_bundle"
    bundle = dossier.to_bundle(bundle_dir)
    expected_names = {
        "dossier.json",
        "dossier_en.html",
        "dossier_es.html",
        "dossier_en.md",
        "dossier_es.md",
        "dossier_en.tex",
        "dossier_es.tex",
        "manifest.json",
    }
    assert {p.name for p in bundle_dir.iterdir()} == expected_names

    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["data_sha256"] == dossier.preflight.data_sha256
    assert set(manifest["checksums"]) == expected_names - {"manifest.json"}
    for fname, recorded_sha in manifest["checksums"].items():
        actual_sha = hashlib.sha256((bundle_dir / fname).read_bytes()).hexdigest()
        assert actual_sha == recorded_sha
        assert "/" not in fname and "\\" not in fname

    # Verify zero host path leakage across every written file in the bundle
    forbidden_fragments = [
        str(tmp_path),
        str(bundle_dir),
        tmp_path.name,
        "secret_host_dir",
        "export_bundle",
        str(Path.home()),
    ]
    for fname in expected_names:
        text = (bundle_dir / fname).read_text(encoding="utf-8")
        for frag in forbidden_fragments:
            if frag and len(frag) > 3:
                assert frag not in text, f"Found host path fragment {frag!r} in {fname}"

    # Verify LaTeX & Markdown bilingual contents contain no error placeholders
    tex_es = (bundle_dir / "dossier_es.tex").read_text(encoding="utf-8")
    assert "\\section*{" in tex_es
    assert "\\subsection*{" in tex_es

    with pytest.raises(FileExistsError):
        dossier.to_bundle(bundle_dir, overwrite=False)
