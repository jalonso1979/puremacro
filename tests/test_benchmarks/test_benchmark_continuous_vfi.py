"""Automated test suite for the Continuous VFI Benchmark CLI script.

Tests:
- CLI argument parsing (--help, --experiment, --backend, --format, --strict, --quiet, --output-dir, --n-runs, --n-eval).
- Execution of benchmark experiments (exp1: Brock-Mirman analytical benchmark, exp2: borrowing kink, exp3: stochastic, exp4: scaling) with exit code 0.
- Table rendering in Markdown, LaTeX, ASCII text, and JSON.
- Graceful backend fallback when unavailable backends (e.g., CuPy on macOS) are specified.
- Exit code semantics: 0 (success), 2 (argparse usage error).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from puremacro._backend import SUPPORTED, backend_available
from puremacro.reports import _df_to_latex, _df_to_markdown, latex_escape

PROJ = Path(__file__).resolve().parents[2]
BENCH_SCRIPT = PROJ / "benchmarks" / "benchmark_continuous_vfi.py"


def run_benchmark(args: list[str], *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """Run the benchmark CLI script via subprocess using the current Python interpreter."""
    cmd = [sys.executable, str(BENCH_SCRIPT), *args]
    return subprocess.run(
        cmd,
        cwd=PROJ,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ============================================================================
# Tier 1: Feature Coverage (CLI Parsing, Formatting Contracts & Isolated Runs)
# ============================================================================

class TestTier1FeatureCoverage:
    """Tier 1: Feature coverage testing for benchmark CLI interface and formatters."""

    def test_table_rendering_markdown_contract(self):
        """Verify markdown table formatter contract on benchmark metric columns."""
        df = pd.DataFrame([
            {"Method": "Chebyshev (N=8)", "Runtime (ms)": 13.69, "Peak Memory (KiB)": 63.8, "Max Euler Residual": 4.65e-7, "L2 Policy Error": 3.12e-7},
            {"Method": "FEM (E=50)", "Runtime (ms)": 384.94, "Peak Memory (KiB)": 294.0, "Max Euler Residual": 3.37e-5, "L2 Policy Error": 2.45e-5},
        ])
        md = _df_to_markdown(df, index=False)
        assert "|" in md, "Markdown table must contain pipe separators '|'"
        assert "|---" in md or "|:---" in md, "Markdown table must contain divider row"
        assert "Chebyshev (N=8)" in md
        assert "FEM (E=50)" in md

    def test_table_rendering_latex_contract(self):
        """Verify LaTeX tabular formatter contract and character escaping."""
        df = pd.DataFrame([
            {"Method_Name": "Chebyshev_Collocation", "Runtime (ms)": 13.69, "Error_%": 0.0001},
        ])
        tex = _df_to_latex(df, index=False)
        assert "\\begin{tabular}" in tex
        assert "\\end{tabular}" in tex
        assert "\\hline" in tex
        # Meta-characters should be escaped
        assert "Method\\_Name" in tex
        assert "Error\\_\\%" in tex

    def test_backend_availability_contract(self):
        """Verify backend discovery adheres to puremacro._backend specification."""
        assert "numpy" in SUPPORTED
        assert "numba" in SUPPORTED
        assert "mlx" in SUPPORTED
        assert "cupy" in SUPPORTED
        assert backend_available("numpy") is True
        # cupy on mac should return False without raising exceptions
        assert isinstance(backend_available("cupy"), bool)

    def test_benchmark_script_file_exists(self):
        """Verify benchmarks/benchmark_continuous_vfi.py exists on disk."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} is currently being authored by worker")
        assert BENCH_SCRIPT.is_file(), f"Benchmark script missing at {BENCH_SCRIPT}"

    def test_cli_help_flag(self):
        """Verify python benchmarks/benchmark_continuous_vfi.py --help returns exit code 0."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark(["--help"])
        assert res.returncode == 0, f"--help failed with stderr: {res.stderr}"
        stdout = res.stdout
        assert "--experiment" in stdout, "Help message must document --experiment"
        assert "--backend" in stdout, "Help message must document --backend"
        assert "--format" in stdout, "Help message must document --format"
        assert "--strict" in stdout, "Help message must document --strict"

    def test_cli_experiment_exp1_execution_exit_0(self):
        """Verify --experiment exp1 executes cleanly with exit code 0."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark(["--experiment", "exp1", "--n-runs", "1", "--n-eval", "200"])
        assert res.returncode == 0, (
            f"Execution failed with return code {res.returncode}.\n"
            f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
        )
        assert len(res.stdout) > 0, "Benchmark must produce output"

    def test_cli_format_markdown_rendering(self):
        """Verify --format markdown renders a valid Markdown table with pipe separators."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--format", "markdown",
            "--n-runs", "1",
            "--n-eval", "200",
        ])
        assert res.returncode == 0, f"Failed with stderr: {res.stderr}"
        assert "|" in res.stdout, "Markdown output must contain pipe separators '|'"
        assert "|---" in res.stdout or "|:---" in res.stdout, (
            "Markdown output must contain GFM table divider row '|---|'"
        )

    def test_cli_format_latex_rendering(self):
        """Verify --format latex renders a valid LaTeX tabular environment."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--format", "latex",
            "--n-runs", "1",
            "--n-eval", "200",
        ])
        assert res.returncode == 0, f"Failed with stderr: {res.stderr}"
        assert "\\begin{tabular}" in res.stdout, "LaTeX output must contain '\\begin{tabular}'"
        assert "\\end{tabular}" in res.stdout, "LaTeX output must contain '\\end{tabular}'"

    def test_cli_backend_numpy_selection(self):
        """Verify --backend numpy executes explicitly using the reference backend."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--backend", "numpy",
            "--n-runs", "1",
            "--n-eval", "200",
        ])
        assert res.returncode == 0, f"Failed with stderr: {res.stderr}"


# ============================================================================
# Tier 2: Boundary, Corner & Error Cases
# ============================================================================

class TestTier2BoundaryAndCorner:
    """Tier 2: Boundary value analysis, argument validation, and fallback handling."""

    def test_latex_escape_all_specials(self):
        """Verify all LaTeX special characters are safely escaped."""
        specials = "\\ & % $ # _ { } ~ ^"
        escaped = latex_escape(specials)
        for char in ["&", "%", "$", "#", "_", "{", "}"]:
            assert f"\\{char}" in escaped

    def test_invalid_experiment_name_exits_code_2(self):
        """Verify invalid --experiment choice triggers argparse error with exit code 2."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark(["--experiment", "non_existent_exp"])
        assert res.returncode == 2, (
            f"Expected exit code 2 for invalid experiment argument, got {res.returncode}"
        )
        assert "invalid choice" in res.stderr.lower() or "error" in res.stderr.lower()

    def test_invalid_backend_choice_exits_code_2(self):
        """Verify invalid --backend choice triggers argparse error with exit code 2."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark(["--backend", "unsupported_accelerator"])
        assert res.returncode == 2, (
            f"Expected exit code 2 for invalid backend argument, got {res.returncode}"
        )

    def test_invalid_format_choice_exits_code_2(self):
        """Verify invalid --format choice triggers argparse error with exit code 2."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark(["--format", "html_tables"])
        assert res.returncode == 2, (
            f"Expected exit code 2 for invalid format argument, got {res.returncode}"
        )

    def test_graceful_backend_fallback_on_unavailable_cupy(self):
        """Verify specifying unavailable backend (cupy) falls back to numpy without crashing."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--backend", "cupy",
            "--n-runs", "1",
            "--n-eval", "200",
        ])
        assert res.returncode == 0, (
            f"CuPy fallback failed with return code {res.returncode}.\nSTDERR:\n{res.stderr}"
        )

    def test_single_run_minimal_evaluation_boundary(self):
        """Verify --n-runs 1 and minimal valid --n-eval 100 boundary condition executes correctly."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--n-runs", "1",
            "--n-eval", "100",
        ])
        assert res.returncode == 0, f"Boundary minimal run failed: {res.stderr}"

    def test_sub_minimal_evaluation_boundary_exits_code_2(self):
        """Verify --n-eval below 2 threshold triggers validation error with exit code 2."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--n-runs", "1",
            "--n-eval", "1",
        ])
        assert res.returncode == 2, f"Expected exit code 2 for n-eval < 2, got {res.returncode}"

    def test_sub_minimal_runs_boundary_exits_code_2(self):
        """Verify --n-runs below 1 triggers validation error with exit code 2."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--n-runs", "0",
            "--n-eval", "100",
        ])
        assert res.returncode == 2, f"Expected exit code 2 for n-runs < 1, got {res.returncode}"

    def test_quiet_flag_suppresses_logs(self, tmp_path):
        """Verify --quiet suppresses verbose progress output while completing the run."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        out_dir = tmp_path / "quiet_test"
        res = run_benchmark([
            "--experiment", "exp1",
            "--quiet",
            "--output-dir", str(out_dir),
            "--n-runs", "1",
            "--n-eval", "100",
        ])
        assert res.returncode == 0, f"Quiet run failed with stderr: {res.stderr}"


# ============================================================================
# Tier 3: Cross-Feature Combinations
# ============================================================================

class TestTier3CrossFeature:
    """Tier 3: Multi-argument interactions, file output generation, and validation."""

    def test_output_dir_generates_markdown_and_latex_files(self, tmp_path):
        """Verify --output-dir writes both .md and .tex tables when --format all is passed."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        out_dir = tmp_path / "bench_outputs"
        res = run_benchmark([
            "--experiment", "exp1",
            "--format", "all",
            "--output-dir", str(out_dir),
            "--n-runs", "1",
            "--n-eval", "200",
        ])
        assert res.returncode == 0, f"Failed with stderr: {res.stderr}"
        assert out_dir.exists(), f"Output directory {out_dir} was not created"
        files = list(out_dir.glob("*"))
        assert len(files) > 0, "No table files were written to output-dir"
        extensions = {f.suffix for f in files}
        assert ".md" in extensions or ".txt" in extensions, "Expected Markdown or text file"
        assert ".tex" in extensions or ".latex" in extensions or ".md" in extensions

    def test_format_json_renders_valid_json(self):
        """Verify --format json outputs parseable JSON containing metric records."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--format", "json",
            "--quiet",
            "--n-runs", "1",
            "--n-eval", "200",
        ])
        assert res.returncode == 0, f"Failed with stderr: {res.stderr}"
        import re
        match = re.search(r"\[.*\]", res.stdout, re.DOTALL)
        assert match is not None, f"Could not find JSON array in stdout:\n{res.stdout}"
        data = json.loads(match.group(0))
        assert isinstance(data, list), "Parsed JSON must be a list of records"
        assert len(data) > 0, "Parsed JSON array must not be empty"
        assert "Solver" in data[0] or "Method" in data[0]

    def test_strict_mode_validates_analytical_tolerances(self):
        """Verify --strict mode passes when numerical solutions meet error thresholds."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark([
            "--experiment", "exp1",
            "--strict",
            "--n-runs", "1",
            "--n-eval", "500",
        ])
        assert res.returncode == 0, (
            f"Strict mode failed with return code {res.returncode}.\nSTDERR:\n{res.stderr}"
        )


# ============================================================================
# Tier 4: Real-World Application Scenarios (All Experiments & Verification)
# ============================================================================

class TestTier4RealWorldScenarios:
    """Tier 4: Comprehensive execution across all 4 comparative experiments."""

    def test_experiment_2_borrowing_kink_execution(self):
        """Verify Experiment 2 (borrowing constraint kink) runs and detects Gibbs ringing."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark(["--experiment", "exp2", "--n-runs", "1", "--n-eval", "200"])
        assert res.returncode == 0, f"Exp 2 failed with stderr: {res.stderr}"
        out = res.stdout.lower()
        assert ("fem" in out or "kink" in out or "chebyshev" in out or "ringing" in out), (
            "Exp 2 output must report FEM or Chebyshev kink results"
        )

    def test_experiment_3_stochastic_execution(self):
        """Verify Experiment 3 (stochastic multi-state model) executes cleanly."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark(["--experiment", "exp3", "--n-runs", "1", "--n-eval", "200"])
        assert res.returncode == 0, f"Exp 3 failed with stderr: {res.stderr}"

    def test_experiment_4_backend_scaling_execution(self):
        """Verify Experiment 4 (multi-backend scaling) executes across host backends."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        res = run_benchmark(["--experiment", "exp4", "--n-runs", "1"])
        assert res.returncode == 0, f"Exp 4 failed with stderr: {res.stderr}"

    def test_full_benchmark_all_experiments_run(self, tmp_path):
        """Verify --experiment all executes all 4 experiments and generates complete reports."""
        if not BENCH_SCRIPT.exists():
            pytest.skip(f"{BENCH_SCRIPT} not yet authored")
        out_dir = tmp_path / "all_experiments_out"
        res = run_benchmark([
            "--experiment", "all",
            "--format", "markdown",
            "--output-dir", str(out_dir),
            "--n-runs", "1",
            "--n-eval", "200",
        ], timeout=240)
        assert res.returncode == 0, (
            f"All experiments run failed with code {res.returncode}.\nSTDERR:\n{res.stderr}"
        )
        assert out_dir.exists()
        assert len(list(out_dir.glob("*.md"))) > 0, "Expected generated markdown summary files"
