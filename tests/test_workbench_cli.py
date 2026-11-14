"""Unit and CLI integration tests for puremacro.workbench and puremacro.cli."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import puremacro.cli as pm_cli
from puremacro.dsge import (
    IRFScorecardResult,
    NormalPrior,
    PriorPosteriorComparisonResult,
    nuts_sample,
)
from puremacro.solvers.diagnostics import (
    AccuracyStatus,
    ConvergenceStatus,
    SolverDiagnosticRecord,
    SupportStatus,
    VALID_OVERALL_STATUSES,
)
from puremacro.structural.bridge import MomentTargets
from puremacro.workbench import (
    EmpiricalStructuralDossier,
    run_empirical_structural_workbench,
)


REPO_ROOT = Path(__file__).resolve().parents[1]

_PYODIDE_ALLOWED_TOP_LEVEL = frozenset(
    set(sys.stdlib_module_names)
    | {"__future__", "numpy", "scipy", "pandas", "matplotlib", "requests"}
)
_PYODIDE_FORBIDDEN_STDLIB = frozenset({"sqlite3", "ssl"})


def _make_ts_df(
    n: int = 84,
    *,
    seed: int = 42,
    freq: str = "QS",
    as_date_col: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-01", periods=n, freq=freq)
    x = np.zeros(n)
    z = np.zeros(n)
    y = np.zeros(n)
    eps_x = rng.normal(0.0, 0.8, size=n)
    eps_z = rng.normal(0.0, 0.5, size=n)
    eps_y = rng.normal(0.0, 0.35, size=n)
    for t in range(1, n):
        z[t] = 0.55 * z[t - 1] + eps_z[t]
        x[t] = 0.45 * x[t - 1] + eps_x[t]
        y[t] = 0.40 * y[t - 1] - 0.80 * x[t] - 0.50 * x[t - 1] + 0.30 * z[t - 1] + eps_y[t]
    df = pd.DataFrame({"y": y, "x": x, "z": z}, index=idx)
    if as_date_col:
        out = df.reset_index().rename(columns={"index": "date"})
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        return out
    return df


def _make_panel_df(
    n_units: int = 6,
    n_periods: int = 24,
    *,
    seed: int = 101,
    multiindex: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2012-01-01", periods=n_periods, freq="QS")
    rows: list[dict[str, object]] = []
    for u in range(n_units):
        iso = f"U{u:02d}"
        fe = float(u - n_units / 2.0) * 0.15
        x_vals = np.zeros(n_periods)
        y_vals = np.zeros(n_periods)
        z_vals = np.zeros(n_periods)
        for t in range(1, n_periods):
            z_vals[t] = 0.4 * z_vals[t - 1] + rng.normal(0.0, 0.4)
            x_vals[t] = 0.35 * x_vals[t - 1] + rng.normal(0.0, 0.8)
            y_vals[t] = (
                fe
                + 0.3 * y_vals[t - 1]
                - 0.65 * x_vals[t]
                - 0.35 * x_vals[t - 1]
                + 0.25 * z_vals[t - 1]
                + rng.normal(0.0, 0.3)
            )
        for t in range(n_periods):
            rows.append(
                {
                    "country": iso,
                    "date": dates[t].strftime("%Y-%m-%d"),
                    "y": float(y_vals[t]),
                    "x": float(x_vals[t]),
                    "z": float(z_vals[t]),
                }
            )
    df = pd.DataFrame(rows)
    if multiindex:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["country", "date"]).sort_index()
    return df


def _make_irf_df(
    horizons: list[int] | None = None,
    *,
    peak: float = -0.8,
    se: float = 0.2,
    y_col: str = "point",
    variable: str | None = None,
) -> pd.DataFrame:
    hs = horizons if horizons is not None else [0, 1, 2, 3, 4]
    pts = np.array([peak * (0.7 ** h) for h in hs], dtype=float)
    se_arr = np.full(len(hs), se, dtype=float)
    data: dict[str, object] = {
        "h": hs,
        y_col: pts,
        "se": se_arr,
        "lower": pts - 1.645 * se_arr,
        "upper": pts + 1.645 * se_arr,
    }
    if variable is not None:
        data["variable"] = [variable] * len(hs)
    return pd.DataFrame(data)


def _extract_top_level_import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            roots.add(node.module.split(".", 1)[0])
    return roots


class _HANKLikeResult:
    """Minimal HANK-shaped container exercising workbench HANK adaptation."""

    def __init__(self, target: str = "y", horizons: int = 5) -> None:
        self.model_name = "HANK-Toy"
        self.shock_name = "eps_mp"
        self.converged = True
        self.steady_state = {"Y": 1.0, "r": 0.01}
        self.transition_paths = pd.DataFrame(
            {
                "h": np.arange(horizons, dtype=int),
                target: np.array([-0.8 * (0.7 ** h) for h in range(horizons)], dtype=float),
            }
        )
        self.diagnostics = SolverDiagnosticRecord(
            convergence=ConvergenceStatus(
                converged=True,
                iterations=14,
                stopping_reason="tol_reached",
                solver_method="sequence_space_jacobian",
                tolerance=1e-9,
            ),
            accuracy=AccuracyStatus(
                is_accurate=True,
                max_residual=2.5e-8,
                residual_norm=4.1e-8,
                residual_type="nk_residual",
                tolerance=1e-5,
            ),
            support=SupportStatus(is_supported=True, regime="interior"),
            overall_status="PASS",
        )


# ---------------------------------------------------------------------------
# 1. Pyodide top-level & lazy-import invariants
# ---------------------------------------------------------------------------


def test_top_level_imports_pyodide_allowlist() -> None:
    """Module top-level imports in workbench and cli obey the Pyodide allowlist."""
    for rel in ("puremacro/workbench.py", "puremacro/cli.py"):
        roots = _extract_top_level_import_roots(REPO_ROOT / rel)
        disallowed = roots - _PYODIDE_ALLOWED_TOP_LEVEL
        assert not disallowed, f"{rel} has disallowed top-level imports: {sorted(disallowed)}"
        forbidden_std = roots & _PYODIDE_FORBIDDEN_STDLIB
        assert not forbidden_std, f"{rel} imports forbidden Pyodide stdlib modules: {sorted(forbidden_std)}"


def test_lazy_import_isolation_in_fresh_subprocess() -> None:
    """Importing puremacro.workbench and puremacro.cli avoids heavy submodules eagerly."""
    must_not_be_eager = {
        "puremacro.lp",
        "puremacro.dsge",
        "puremacro.fetch",
        "puremacro.reports",
        "puremacro.structural",
        "puremacro.solvers",
        "statsmodels",
        "linearmodels",
        "arch",
        "bs4",
        "pdfplumber",
        "pypdf",
        "openpyxl",
        "sqlite3",
        "ssl",
    }

    # puremacro.cli does not eagerly import puremacro.workbench or heavy submodules
    cli_probe = (
        "import json, sys, puremacro.cli; "
        "print(json.dumps(sorted(sys.modules)))"
    )
    cli_proc = subprocess.run(
        [sys.executable, "-c", cli_probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    cli_loaded = set(json.loads(cli_proc.stdout))
    assert "puremacro.workbench" not in cli_loaded
    assert not (cli_loaded & must_not_be_eager)

    probe = (
        "import json, sys, puremacro.workbench, puremacro.cli; "
        "loaded = sorted(m for m in sys.modules); "
        "print(json.dumps(loaded))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    loaded = set(json.loads(proc.stdout))
    eager_hits = loaded & must_not_be_eager
    assert not eager_hits, f"Eagerly imported banned/heavy modules: {sorted(eager_hits)}"


# ---------------------------------------------------------------------------
# 2. Workbench pipeline happy paths & exports
# ---------------------------------------------------------------------------


def test_workbench_time_series_happy_path_and_frozen_dossier() -> None:
    """Single time series with DatetimeIndex runs full empirical + moment-target pipeline."""
    df = _make_ts_df(n=84, seed=42, freq="QS")
    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        controls=("z",),
        horizons=(0, 1, 2, 3, 4),
        lags=2,
        ci=0.90,
        freq="Q",
        response_unit="percent",
        shock_unit="100bp",
    )

    assert isinstance(dossier, EmpiricalStructuralDossier)
    assert dossier.passed is True
    assert dossier.overall_status in {"PASS", "PASS_WITH_WARNINGS"}
    assert dossier.preflight.passed is True
    assert list(dossier.empirical_lp["h"]) == [0, 1, 2, 3, 4]
    # Negative shock coefficient on impact should be close to -0.8
    assert float(dossier.empirical_lp.loc[dossier.empirical_lp["h"] == 0, "beta"].iloc[0]) < -0.3

    assert dossier.sensitivity is not None
    assert len(dossier.sensitivity.metrics) >= 2
    assert len(dossier.irf_comparison.metrics) >= 1
    assert isinstance(dossier.moment_targets, MomentTargets)
    assert len(dossier.moment_targets.values) == 5
    assert dossier.moment_targets.covariance.shape == (5, 5)
    assert math.isfinite(float(dossier.moment_targets.values[0]))
    assert dossier.irf_scorecard is None
    assert dossier.prior_posterior is None

    # Frozen dataclass immutability
    with pytest.raises(FrozenInstanceError):
        dossier.lags = 5  # type: ignore[misc]


def test_workbench_numeric_and_date_column_handling() -> None:
    """Integer time columns and CSV-style date columns execute without nanosecond coercion."""
    df_csv = _make_ts_df(n=72, seed=7, freq="QS", as_date_col=True)
    dossier_date = run_empirical_structural_workbench(
        df_csv,
        "y",
        "x",
        time_col="date",
        horizons=(0, 1, 2, 3),
        lags=2,
        run_sensitivity=False,
    )
    assert dossier_date.preflight.passed is True
    assert isinstance(dossier_date.moment_targets, MomentTargets)
    assert dossier_date.sensitivity is None

    # Numeric integer time index should stay numeric rather than turning into 1970-01-01 ns timestamps
    df_num = _make_ts_df(n=72, seed=9, freq="QS").reset_index(drop=True)
    df_num["t_idx"] = np.arange(1, len(df_num) + 1, dtype=int)
    dossier_num = run_empirical_structural_workbench(
        df_num,
        "y",
        "x",
        time_col="t_idx",
        horizons=(0, 1, 2, 3),
        lags=1,
        run_sensitivity=False,
        build_moment_targets=False,
    )
    assert dossier_num.preflight.passed is True
    assert len(dossier_num.empirical_lp) == 4


def test_workbench_panel_and_multiindex_workflows() -> None:
    """Balanced long panel and MultiIndex panel run panel_lp and lag sensitivity cleanly."""
    panel_df = _make_panel_df(n_units=5, n_periods=22, seed=11, multiindex=False)
    dossier = run_empirical_structural_workbench(
        panel_df,
        "y",
        "x",
        unit_col="country",
        time_col="date",
        controls=("z",),
        horizons=(0, 1, 2, 3),
        lags=1,
        strict_preflight=False,
    )
    assert dossier.metadata["is_panel"] is True
    assert dossier.metadata["estimator"] == "panel_lp"
    assert dossier.sensitivity is not None
    assert len(dossier.sensitivity.metrics) >= 2
    assert dossier.moment_targets is None

    mi_df = _make_panel_df(n_units=5, n_periods=22, seed=12, multiindex=True)
    dossier_mi = run_empirical_structural_workbench(
        mi_df,
        "y",
        "x",
        unit_col="country",
        time_col="date",
        horizons=(0, 1, 2),
        lags=1,
        run_sensitivity=False,
        strict_preflight=False,
    )
    assert dossier_mi.metadata["is_panel"] is True
    assert len(dossier_mi.empirical_lp) == 3


def test_workbench_structural_scorecard_prior_posterior_and_extra_irfs() -> None:
    """Full structural loop scores model_irf, evaluates prior-posterior, and compares extra_irfs."""
    df = _make_ts_df(n=80, seed=55, freq="QS")

    # Quick conjugate NUTS chain for PriorPosteriorComparisonResult
    def log_post(theta: np.ndarray) -> tuple[float, np.ndarray]:
        mu = float(theta[0])
        lp = -0.5 * ((mu - 0.5) / 0.25) ** 2
        grad = -((mu - 0.5) / (0.25 ** 2))
        return lp, np.array([grad], dtype=float)

    post = nuts_sample(
        log_post,
        init_params=np.array([0.5]),
        n_draws=60,
        n_chains=2,
        warmup=40,
        param_names=("rho_r",),
        seed=5,
        priors={"rho_r": NormalPrior(mean=0.5, std=0.5)},
    )

    hank_like = _HANKLikeResult(target="y", horizons=5)
    extra = _make_irf_df([0, 1, 2, 3, 4], peak=-0.75, se=0.18)

    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        controls=("z",),
        horizons=(0, 1, 2, 3, 4),
        lags=2,
        extra_irfs={"Alt LP": extra},
        model_irf=hank_like,
        model_label="HANK Model",
        posterior=post,
        solver_result=hank_like,
    )

    assert isinstance(dossier.irf_scorecard, IRFScorecardResult)
    assert math.isfinite(dossier.irf_scorecard.rmse)
    assert isinstance(dossier.prior_posterior, PriorPosteriorComparisonResult)
    assert "rho_r" in dossier.prior_posterior.table["param"].tolist()
    cmp_specs = set(dossier.irf_comparison.metrics["spec"].tolist())
    assert "HANK Model" in cmp_specs
    assert "Alt LP" in cmp_specs
    assert dossier.solver_diagnostics is not None
    assert dossier.convergence_status is not None and dossier.convergence_status.converged is True
    assert dossier.accuracy_status is not None and dossier.accuracy_status.is_accurate is True


# ---------------------------------------------------------------------------
# 3. Solver diagnostics status normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("conv_ok", "acc_ok", "supported", "expected_status", "expected_passed"),
    [
        (True, True, True, "PASS", True),
        (True, False, True, "CONVERGED_INACCURATE", False),
        (True, True, False, "UNSUPPORTED_REGIME", False),
        (False, False, True, "FAILED", False),
    ],
)
def test_workbench_solver_diagnostics_status_propagation(
    conv_ok: bool,
    acc_ok: bool,
    supported: bool,
    expected_status: str,
    expected_passed: bool,
) -> None:
    """SolverDiagnosticRecord and tuple/mapping inputs map to canonical overall statuses."""
    assert expected_status in VALID_OVERALL_STATUSES
    df = _make_ts_df(n=64, seed=19, freq="QS")

    diag = SolverDiagnosticRecord(
        convergence=ConvergenceStatus(
            converged=conv_ok,
            iterations=20,
            stopping_reason="done" if conv_ok else "max_iter",
            solver_method="newton",
            tolerance=1e-6,
        ),
        accuracy=AccuracyStatus(
            is_accurate=acc_ok,
            max_residual=1e-7 if acc_ok else 0.05,
            residual_norm=1e-7 if acc_ok else 0.08,
            residual_type="euler_max",
            tolerance=1e-4,
        ),
        support=SupportStatus(
            is_supported=supported,
            regime="interior" if supported else "boundary_zlb",
            warnings=() if supported else ("Near ZLB constraint",),
        ),
        overall_status=expected_status,
    )

    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        horizons=(0, 1, 2),
        lags=1,
        run_sensitivity=False,
        solver_diagnostics=diag,
    )
    assert dossier.overall_status == expected_status
    assert dossier.passed is expected_passed

    # Also test (ConvergenceStatus, AccuracyStatus) tuple coercion and mapping coercion
    if supported:
        dossier_tup = run_empirical_structural_workbench(
            df,
            "y",
            "x",
            horizons=(0, 1, 2),
            lags=1,
            run_sensitivity=False,
            solver_diagnostics=(diag.convergence, diag.accuracy),
        )
        assert dossier_tup.overall_status == expected_status

        dossier_map = run_empirical_structural_workbench(
            df,
            "y",
            "x",
            horizons=(0, 1, 2),
            lags=1,
            run_sensitivity=False,
            solver_diagnostics={
                "converged": conv_ok,
                "iterations": 10,
                "max_residual": 1e-7 if acc_ok else 0.05,
                "is_accurate": acc_ok,
            },
        )
        assert dossier_map.overall_status == expected_status


# ---------------------------------------------------------------------------
# 4. Bilingual summaries, DataFrame views & Markdown/LaTeX/Typst/JSON exports
# ---------------------------------------------------------------------------


def test_dossier_bilingual_and_export_formats() -> None:
    """Verify en/es summaries, all to_frame sections, Markdown/LaTeX/Typst escaping, and strict JSON."""
    df = _make_ts_df(n=76, seed=31, freq="QS")
    model_df = _make_irf_df([0, 1, 2, 3], peak=-0.7, se=0.15, variable="y")

    dossier = run_empirical_structural_workbench(
        df,
        "y",
        "x",
        controls=("z",),
        horizons=(0, 1, 2, 3),
        lags=2,
        model_irf=model_df,
        lang="en",
        title="Shock & Policy (100% _core_ #1)",
    )

    en_sum = dossier.summary("en")
    es_sum = dossier.summary("es")
    assert "- Preflight:" in en_sum
    assert "- Empirical LP & comparison:" in en_sum
    assert "- Diagnóstico previo:" in es_sum
    assert "- LP empírica y comparación:" in es_sum
    with pytest.raises(ValueError, match="lang"):
        dossier.summary("fr")

    for section in ("overview", "preflight", "irf_metrics", "irf_table", "scorecard", "scorecard_metrics"):
        frame = dossier.to_frame(section)
        assert isinstance(frame, pd.DataFrame)
        assert not frame.empty
    assert dossier.to_dataframe("overview").equals(dossier.to_frame("overview"))
    with pytest.raises(ValueError, match="prior-posterior"):
        dossier.to_frame("prior_posterior")

    with pytest.raises(ValueError, match="Unknown section"):
        dossier.to_frame("nonexistent")

    md_es = dossier.to_markdown(lang="es")
    assert "## Resumen del flujo" in md_es
    assert "## Evaluación estructural IRF" in md_es

    tex = dossier.to_latex(lang="en")
    assert r"\begin{tabular}" in tex
    assert r"\end{tabular}" in tex
    assert r"Shock \& Policy (100\% \_core\_ \#1)" in tex

    typ = dossier.to_typst(lang="es")
    assert "#table(" in typ
    assert r"100% \_core\_ \#1" in typ

    raw_json = dossier.to_json()
    parsed = json.loads(raw_json)
    assert parsed["specification"]["y"] == "y"
    assert parsed["specification"]["x"] == "x"
    assert parsed["passed"] is True
    assert "preflight" in parsed
    assert "irf_scorecard" in parsed


# ---------------------------------------------------------------------------
# 5. Validation & error handling in workbench
# ---------------------------------------------------------------------------


def test_workbench_invalid_inputs_and_strict_preflight() -> None:
    """Validate guardrails for bad inputs and strict_preflight=True vs False."""
    df = _make_ts_df(n=60, seed=14, freq="QS")

    with pytest.raises(TypeError, match="DataFrame"):
        run_empirical_structural_workbench({"y": [1, 2]}, "y", "x")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="empty"):
        run_empirical_structural_workbench(df.iloc[:0], "y", "x")

    with pytest.raises(ValueError, match="not found"):
        run_empirical_structural_workbench(df, "missing_y", "x")

    with pytest.raises(ValueError, match="not found"):
        run_empirical_structural_workbench(df, "y", "missing_x")

    with pytest.raises(ValueError, match="Control"):
        run_empirical_structural_workbench(df, "y", "x", controls=("missing_ctrl",))

    with pytest.raises(ValueError, match="lang"):
        run_empirical_structural_workbench(df, "y", "x", lang="de")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lags"):
        run_empirical_structural_workbench(df, "y", "x", lags=-1)

    with pytest.raises(ValueError, match="ci"):
        run_empirical_structural_workbench(df, "y", "x", ci=1.0)

    with pytest.raises(ValueError, match="unique"):
        run_empirical_structural_workbench(df, "y", "x", horizons=(0, 1, 1))

    # Inf in data causes preflight error -> strict_preflight=True raises, False continues cleanly
    bad_df = df.copy()
    bad_df.iloc[10, bad_df.columns.get_loc("y")] = np.inf
    with pytest.raises(ValueError, match="(?i)preflight failed"):
        run_empirical_structural_workbench(bad_df, "y", "x", strict_preflight=True)

    # Also exercise clean marginal fallback warning when NaN blocks joint moment targets
    nan_df = df.copy()
    nan_df.iloc[25, nan_df.columns.get_loc("z")] = np.nan
    dossier_nan = run_empirical_structural_workbench(
        nan_df,
        "y",
        "x",
        controls=("z",),
        horizons=(0, 1, 2),
        lags=1,
        strict_preflight=False,
        run_sensitivity=False,
    )
    assert dossier_nan.preflight.passed is True
    assert dossier_nan.moment_targets is None
    assert any("Joint moment targets unavailable" in w for w in dossier_nan.warnings)


# ---------------------------------------------------------------------------
# 6. CLI subcommands & exit codes
# ---------------------------------------------------------------------------


def test_cli_help_and_doctor_in_subprocess_and_direct() -> None:
    """Verify python -m puremacro.cli --help and doctor subcommand formats."""
    help_proc = subprocess.run(
        [sys.executable, "-m", "puremacro.cli", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_proc.returncode == 0
    assert "preflight" in help_proc.stdout
    assert "scorecard" in help_proc.stdout

    # Direct main(['doctor'])
    out_buf: list[str] = []
    # Use --output file to inspect cleanly alongside stdout capture
    rc = pm_cli.main(["doctor", "--format", "json"])
    assert rc == 0

    proc_doc = subprocess.run(
        [sys.executable, "-m", "puremacro.cli", "doctor", "--format", "json"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    doc_payload = json.loads(proc_doc.stdout)
    assert doc_payload["status"] in {"ok", "warn"}
    assert not doc_payload["required_failures"]
    out_buf.append(proc_doc.stdout)
    assert len(out_buf) == 1


def test_cli_preflight_subcommand_formats_and_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test preflight CLI across JSON/Markdown, output file, strict, and allow-errors."""
    df = _make_ts_df(n=50, seed=3, freq="QS", as_date_col=True)
    csv_path = tmp_path / "macro.csv"
    df.to_csv(csv_path, index=False)

    rc = pm_cli.main(["preflight", str(csv_path), "--time-col", "date", "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["passed"] is True
    assert payload["n_units"] == 1

    out_md = tmp_path / "preflight_es.md"
    rc_md = pm_cli.main(
        [
            "preflight",
            str(csv_path),
            "--time-col",
            "date",
            "--format",
            "markdown",
            "--lang",
            "es",
            "--output",
            str(out_md),
            "--quiet",
        ]
    )
    assert rc_md == 0
    assert capsys.readouterr().out == ""
    md_text = out_md.read_text(encoding="utf-8")
    assert "# Diagnóstico previo del panel" in md_text

    # File with Inf triggers error -> exit 1 unless --allow-errors
    bad_df = df.copy()
    bad_df.loc[5, "y"] = np.inf
    bad_csv = tmp_path / "bad.csv"
    bad_df.to_csv(bad_csv, index=False)

    assert pm_cli.main(["preflight", str(bad_csv), "--time-col", "date", "--format", "json"]) == 1
    assert (
        pm_cli.main(
            ["preflight", str(bad_csv), "--time-col", "date", "--allow-errors", "--format", "json"]
        )
        == 0
    )
    capsys.readouterr()


def test_cli_lp_grid_and_workbench_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test lp-grid both in grid mode and --workbench mode, plus invalid-arg exit 2."""
    df = _make_ts_df(n=72, seed=21, freq="QS", as_date_col=True)
    csv_path = tmp_path / "ts.csv"
    df.to_csv(csv_path, index=False)

    rc = pm_cli.main(
        [
            "lp-grid",
            str(csv_path),
            "--y",
            "y",
            "--x",
            "x",
            "--controls",
            "z",
            "--time-col",
            "date",
            "--horizons",
            "0..4",
            "--lags",
            "1,2",
            "--estimators",
            "lp_hac,la_lp",
            "--baseline-lags",
            "2",
            "--baseline-estimator",
            "lp_hac",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    grid_json = json.loads(capsys.readouterr().out)
    assert grid_json["horizons"] == [0, 1, 2, 3, 4]
    assert len(grid_json["metrics"]) == 4

    rc_wb = pm_cli.main(
        [
            "lp-grid",
            str(csv_path),
            "--y",
            "y",
            "--x",
            "x",
            "--time-col",
            "date",
            "--horizons",
            "0..3",
            "--lags",
            "1,2",
            "--workbench",
            "--format",
            "markdown",
            "--lang",
            "es",
        ]
    )
    assert rc_wb == 0
    wb_md = capsys.readouterr().out
    assert "# Expediente empírico-estructural" in wb_md

    # Single grid point without --workbench is a CLI usage error (exit 2)
    rc_bad = pm_cli.main(
        [
            "lp-grid",
            str(csv_path),
            "--y",
            "y",
            "--x",
            "x",
            "--lags",
            "2",
        ]
    )
    assert rc_bad == 2
    capsys.readouterr()


def test_cli_compare_irfs_and_scorecard(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test compare-irfs (positional, --spec, --bundle) and scorecard CLI subcommands."""
    irf_a = _make_irf_df([0, 1, 2, 3, 4], peak=-0.8, se=0.2)
    irf_b = _make_irf_df([0, 1, 2, 3, 4], peak=-0.65, se=0.18)
    path_a = tmp_path / "irf_a.csv"
    path_b = tmp_path / "irf_b.csv"
    irf_a.to_csv(path_a, index=False)
    irf_b.to_csv(path_b, index=False)

    rc_cmp = pm_cli.main(
        [
            "compare-irfs",
            f"LP_HAC={path_a}",
            "--spec",
            f"DSGE={path_b}",
            "--baseline",
            "LP_HAC",
            "--format",
            "json",
        ]
    )
    assert rc_cmp == 0
    cmp_json = json.loads(capsys.readouterr().out)
    assert cmp_json["baseline"] == "LP_HAC"
    assert {row["spec"] for row in cmp_json["metrics"]} == {"LP_HAC", "DSGE"}

    # Bundle JSON comparison
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(
        json.dumps(
            {
                "Baseline": irf_a.to_dict(orient="records"),
                "Alt": irf_b.to_dict(orient="records"),
            }
        ),
        encoding="utf-8",
    )
    rc_bun = pm_cli.main(
        ["compare-irfs", "--bundle", str(bundle_path), "--format", "markdown", "--lang", "en"]
    )
    assert rc_bun == 0
    assert "# IRF Comparison" in capsys.readouterr().out

    # Scorecard CLI using MomentTargets JSON and multi-shock model IRF JSON
    cov = np.diag([0.04, 0.04, 0.04, 0.04, 0.04])
    targets_payload = {
        "values": [-0.8 * (0.7 ** h) for h in range(5)],
        "covariance": cov.tolist(),
        "labels": [f"y:h={h}" for h in range(5)],
        "units": "percent",
        "metadata": {"shock": "mp"},
    }
    targets_json = tmp_path / "targets.json"
    targets_json.write_text(json.dumps(targets_payload), encoding="utf-8")

    model_multi = {
        "mp": _make_irf_df([0, 1, 2, 3, 4], peak=-0.75, se=0.1, variable="y").to_dict(
            orient="records"
        )
    }
    model_json = tmp_path / "model.json"
    model_json.write_text(json.dumps(model_multi), encoding="utf-8")

    rc_sc = pm_cli.main(
        [
            "scorecard",
            "--model-irf",
            str(model_json),
            "--targets",
            str(targets_json),
            "--shock",
            "mp",
            "--weight-mode",
            "covariance",
            "--format",
            "json",
        ]
    )
    assert rc_sc == 0
    sc_json = json.loads(capsys.readouterr().out)
    assert sc_json["weight_mode_used"] == "covariance"
    assert sc_json["chi2_df"] == 5
    assert math.isfinite(sc_json["rmse"])

    rc_sc_md = pm_cli.main(
        [
            "scorecard",
            "--model-irf",
            str(model_json),
            "--targets",
            str(targets_json),
            "--format",
            "markdown",
            "--lang",
            "es",
        ]
    )
    assert rc_sc_md == 0
    assert "# Evaluación de IRF estructural frente a objetivos empíricos" in capsys.readouterr().out


def test_cli_freshness_recorded_cache_and_error_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test freshness CLI in offline recorded/cache modes and bad CLI exit codes."""
    rc_rec = pm_cli.main(
        [
            "freshness",
            "--mode",
            "recorded",
            "--sources",
            "fred.csv",
            "--as-of",
            "2025-02-01",
            "--format",
            "json",
        ]
    )
    assert rc_rec == 0
    rec_json = json.loads(capsys.readouterr().out)
    assert rec_json["mode"] == "recorded"
    assert len(rec_json["records"]) == 1
    assert rec_json["records"][0]["source_id"] == "fred.csv"

    # Cache mode backed by nonexistent DB -> clean not_cached status (exit 0)
    absent_db = tmp_path / "absent_cache.sqlite"
    rc_cache = pm_cli.main(
        [
            "freshness",
            "--mode",
            "cache",
            "--sources",
            "fred.csv",
            "--db-path",
            str(absent_db),
            "--format",
            "markdown",
            "--lang",
            "es",
        ]
    )
    assert rc_cache == 0
    assert "fred.csv" in capsys.readouterr().out

    # Seeded SQLite cache DB recording a failed telemetry event -> --fail-on-outage exits 1
    from puremacro import _cache_db

    outage_db = tmp_path / "outage_cache.sqlite"
    con = sqlite3.connect(outage_db)
    try:
        _cache_db.bootstrap_schema(con)
        con.execute(
            "INSERT INTO connector_events (ts, source, outcome, fallback_used) "
            "VALUES (?, ?, ?, ?)",
            (1700000000, "fred", "error", "none"),
        )
        con.commit()
    finally:
        con.close()

    rc_outage = pm_cli.main(
        [
            "freshness",
            "--mode",
            "cache",
            "--sources",
            "fred.csv",
            "--db-path",
            str(outage_db),
            "--fail-on-outage",
            "--format",
            "json",
        ]
    )
    assert rc_outage == 1
    capsys.readouterr()

    # Exit 2 on unknown subcommand or bad syntax, Exit 1 on missing file
    assert pm_cli.main(["unknown-subcmd"]) == 2
    assert pm_cli.main(["preflight", str(tmp_path / "does_not_exist.csv")]) == 1
    capsys.readouterr()


def _make_did_panel_df(
    n_units: int = 8,
    n_periods: int = 20,
    *,
    seed: int = 77,
    multiindex: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=n_periods, freq="QS")
    rows: list[dict[str, object]] = []
    # Half units treated around staggered cohorts, half never-treated clean controls
    cohort_times = [6, 8, 10, 12] + [None] * max(1, n_units - 4)
    for u in range(n_units):
        uid = f"U{u:02d}"
        c_t = cohort_times[u]
        fe = (u - n_units / 2.0) * 0.1
        y_prev = 0.0
        for t in range(n_periods):
            d_val = 1 if (c_t is not None and t >= c_t) else 0
            eff = 1.5 if d_val == 1 else 0.0
            y_val = fe + 0.3 * y_prev + eff + float(rng.normal(0.0, 0.25))
            y_prev = y_val - eff
            rows.append(
                {
                    "unit": uid,
                    "date": dates[t].strftime("%Y-%m-%d"),
                    "y": y_val,
                    "treat": d_val,
                }
            )
    df = pd.DataFrame(rows)
    if multiindex:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index(["unit", "date"]).sort_index()
    return df


def test_workbench_and_cli_lp_did_support(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify lp_did works cleanly in workbench (flat & MultiIndex) and CLI lp-grid."""
    from puremacro.lp.lp_did import LPDiDResult, lp_did

    did_df = _make_did_panel_df(n_units=8, n_periods=20, seed=77, multiindex=False)
    dossier = run_empirical_structural_workbench(
        did_df,
        "y",
        "treat",
        unit_col="unit",
        time_col="date",
        horizons=(0, 1, 2, 3),
        lags=1,
        estimator=lp_did,
        base_kwargs={"pre_window": 2},
        sensitivity_on_error="raise",
    )
    assert isinstance(dossier.empirical_lp, LPDiDResult)
    assert dossier.metadata["estimator"] == "lp_did"
    assert dossier.sensitivity is not None
    assert len(dossier.sensitivity.metrics) >= 2
    assert float(dossier.empirical_lp.estimates.loc[0, "beta"]) > 0.5
    parsed = json.loads(dossier.to_json())
    assert len(parsed["empirical_lp"]) == len(dossier.empirical_lp.estimates)
    assert "## Resumen del flujo" in dossier.to_markdown(lang="es")

    # MultiIndex panel with lp_did flattens cleanly into flat unit/time columns
    did_mi = _make_did_panel_df(n_units=8, n_periods=20, seed=78, multiindex=True)
    dossier_mi = run_empirical_structural_workbench(
        did_mi,
        "y",
        "treat",
        unit_col="unit",
        time_col="date",
        horizons=(0, 1, 2),
        lags=0,
        estimator=lp_did,
        base_kwargs={"pre_window": 2},
        run_sensitivity=False,
    )
    assert isinstance(dossier_mi.empirical_lp, LPDiDResult)

    # CLI lp-grid with --estimators lp_did (both grid mode and --workbench mode)
    did_csv = tmp_path / "did.csv"
    did_df.to_csv(did_csv, index=False)

    rc_grid = pm_cli.main(
        [
            "lp-grid",
            str(did_csv),
            "--y",
            "y",
            "--x",
            "treat",
            "--unit-col",
            "unit",
            "--time-col",
            "date",
            "--estimators",
            "lp_did",
            "--horizons",
            "0..3",
            "--lags",
            "0,1",
            "--format",
            "json",
        ]
    )
    assert rc_grid == 0
    grid_out = json.loads(capsys.readouterr().out)
    assert len(grid_out["metrics"]) == 2

    rc_wb = pm_cli.main(
        [
            "lp-grid",
            str(did_csv),
            "--y",
            "y",
            "--x",
            "treat",
            "--unit-col",
            "unit",
            "--time-col",
            "date",
            "--estimators",
            "lp_did",
            "--horizons",
            "0..3",
            "--lags",
            "0,1",
            "--workbench",
            "--format",
            "json",
        ]
    )
    assert rc_wb == 0
    wb_out = json.loads(capsys.readouterr().out)
    assert wb_out["metadata"]["estimator"] == "lp_did"
    assert wb_out["passed"] is True


def test_single_unit_with_unit_col_and_incompatible_estimator_mixes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verify TS estimators do not receive leaked panel_kwargs and incompatible mixes reject cleanly."""
    from puremacro.lp import lp_hac, lp_did, panel_lp

    # Single-unit slice carrying an explicit 'unit' column auto-detected by preflight_panel
    single_df = _make_ts_df(n=72, seed=91, freq="QS", as_date_col=True)
    single_df.insert(0, "unit", "US")

    dossier_single = run_empirical_structural_workbench(
        single_df,
        "y",
        "x",
        controls=("z",),
        horizons=(0, 1, 2, 3),
        lags=2,
        estimator=lp_hac,
        sensitivity_on_error="raise",
    )
    assert dossier_single.passed is True
    assert dossier_single.sensitivity is not None
    assert len(dossier_single.sensitivity.metrics) >= 2

    single_csv = tmp_path / "single_unit.csv"
    single_df.to_csv(single_csv, index=False)
    rc_single = pm_cli.main(
        [
            "lp-grid",
            str(single_csv),
            "--y",
            "y",
            "--x",
            "x",
            "--unit-col",
            "unit",
            "--time-col",
            "date",
            "--estimators",
            "lp_hac,la_lp",
            "--horizons",
            "0..3",
            "--lags",
            "1,2",
            "--format",
            "json",
        ]
    )
    assert rc_single == 0
    assert len(json.loads(capsys.readouterr().out)["metrics"]) == 4

    # Mixing lp_did with FE/TS estimators must fail explicitly rather than crash mid-grid
    panel_df = _make_panel_df(n_units=4, n_periods=20, seed=92, multiindex=False)
    with pytest.raises(ValueError, match="Incompatible LP estimators"):
        run_empirical_structural_workbench(
            panel_df,
            "y",
            "x",
            unit_col="country",
            time_col="date",
            estimator=panel_lp,
            sensitivity_grid={"estimator": [panel_lp, lp_did], "lags": [1]},
            strict_preflight=False,
        )

    panel_csv = tmp_path / "panel_mix.csv"
    panel_df.to_csv(panel_csv, index=False)

    rc_mix = pm_cli.main(
        [
            "lp-grid",
            str(panel_csv),
            "--y",
            "y",
            "--x",
            "x",
            "--unit-col",
            "country",
            "--time-col",
            "date",
            "--estimators",
            "panel_lp,lp_did",
            "--lags",
            "1",
        ]
    )
    assert rc_mix == 2
    capsys.readouterr()

    # TS estimator on multi-unit panel returns CLI usage error (exit 2)
    rc_multi_ts = pm_cli.main(
        [
            "lp-grid",
            str(panel_csv),
            "--y",
            "y",
            "--x",
            "x",
            "--unit-col",
            "country",
            "--time-col",
            "date",
            "--estimators",
            "lp_hac",
            "--lags",
            "1,2",
        ]
    )
    assert rc_multi_ts == 2
    capsys.readouterr()
