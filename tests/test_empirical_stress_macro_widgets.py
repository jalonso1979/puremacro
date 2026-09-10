"""Adversarial stress test harness for Dynare Macro Preprocessor and Interactive IRF Widgets.

Empirically tests:
1. Macro Preprocessor:
   - Deeply nested loops (3- and 4-level nesting, lexical scoping, variable shadowing)
   - Multi-level conditionals (@#if, @#elseif, @#elif, @#else, @#endif, nested conditionals)
   - Modulo operators (@#if i % 3 == 0, loop filters 'when i % 4 == 0', built-in mod(x, y), zero division)
   - String concatenations (+ with string, int, expressions)
   - Dynamic @#include (variable path, expression path, nested includes, circular detection)
   - Cartesian products ([1, 2] * ['a', 'b'], 3-way products, Cartesian power ^)
   - Destructuring tuples (2-tuple, 3-tuple, error handling on mismatch)
   - Error recovery & syntax diagnostics (unclosed blocks, undefined variables, math errors)
   - Full end-to-end Dynare .mod parsing & solving with macro directives and % comments

2. Interactive IRF Widgets:
   - Rapid consecutive parameter slider adjustments (50 updates on NK model) asserting mean latency < 15ms
   - Rapid consecutive parameter slider adjustments on Smets-Wouters (2007) (50 updates) asserting mean latency < 15ms
   - Parameter dragging across determinacy boundaries (phi_pi: 1.5 -> 0.5 -> 2.0 -> 0.1 -> 1.8 -> 0.0 -> 1.5)
   - Graceful NaN handling during indeterminacy without exceptions or canvas crashes
   - Status warning banner display during indeterminacy and clean clearance upon restoration
   - Exact numerical fidelity of restored IRFs against independent fresh model solves
   - Export methods (.summary, .to_markdown, .to_latex, .to_typst, .to_frame) during indeterminacy
   - Multi-parameter simultaneous updates, reset functionality, and resource cleanup
"""
from __future__ import annotations

from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from puremacro.dsge import (
    build_dynare,
    load_mod,
    preprocess_macro,
    DynareMacroError,
    LinearModel,
)
from puremacro.dsge.widgets import interactive_irf, InteractiveIRFResult


# ==============================================================================
# Model Fixtures
# ==============================================================================

def _nk_equations(lead, curr, lag, shocks, p):
    """Textbook 3-equation New Keynesian model with AR(1) cost-push and monetary shocks."""
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
# 1. Macro Preprocessor Adversarial Stress Tests
# ==============================================================================

class TestMacroPreprocessorStress:
    """Adversarial stress testing of the Dynare Macro Preprocessor."""

    def test_deeply_nested_loops_and_scoping(self):
        """Test 3-level and 4-level nested loops with variable shadowing and scope restoration."""
        text = """
@#define ACCUM = 100
@#for i in 1:2
  @#for j in ["a", "b"]
    @#for k in [10, 20]
      @#for l in [0, 1]
node_@{i}_@{j}_@{k}_@{l} = @{i * k + l};
      @#endfor
    @#endfor
  @#endfor
@#endfor
after_loop = @{ACCUM};
"""
        out = preprocess_macro(text).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        # 2 * 2 * 2 * 2 = 16 generated lines + 1 after_loop
        assert len(lines) == 17
        assert "node_1_a_10_0 = 10;" in lines
        assert "node_1_a_10_1 = 11;" in lines
        assert "node_2_b_20_1 = 41;" in lines
        assert "after_loop = 100;" in lines

    def test_loop_variable_shadowing_and_isolation(self):
        """Verify inner loop shadowing does not corrupt outer loop variable upon exit."""
        text = """
@#for x in [10, 20]
outer_before_@{x}
  @#for x in [1, 2]
  inner_@{x}
  @#endfor
outer_after_@{x}
@#endfor
"""
        out = preprocess_macro(text)
        assert "outer_before_10\n  inner_1\n  inner_2\nouter_after_10" in out
        assert "outer_before_20\n  inner_1\n  inner_2\nouter_after_20" in out

    def test_multi_level_conditionals(self):
        """Test nested conditionals, multi-branch elseif/elif, and complex boolean logic."""
        text = """
@#for val in [-5, 0, 1, 2, 3, 10]
  @#if val < 0
negative: @{val}
  @#elif val == 0
zero
  @#elseif val == 1
one
  @#elseif val == 2
two
  @#else
    @#if val == 3 || val == 4
three_or_four: @{val}
    @#elseif val > 5 && val < 15
between_five_and_fifteen: @{val}
    @#else
other: @{val}
    @#endif
  @#endif
@#endfor
"""
        out = preprocess_macro(text).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        assert lines == [
            "negative: -5",
            "zero",
            "one",
            "two",
            "three_or_four: 3",
            "between_five_and_fifteen: 10",
        ]

    def test_modulo_operators_in_macro(self):
        """Test modulo operator (%) in conditionals, loop filters (when), and expressions."""
        # 1. Modulo in @#if / @#elseif
        text_if = """
@#for i in 1:6
  @#if i % 3 == 0
mult_3: @{i}
  @#elseif i % 3 == 1
mod_1: @{i}
  @#else
mod_2: @{i}
  @#endif
@#endfor
"""
        out_if = preprocess_macro(text_if).strip()
        lines_if = [l.strip() for l in out_if.splitlines() if l.strip()]
        assert lines_if == [
            "mod_1: 1",
            "mod_2: 2",
            "mult_3: 3",
            "mod_1: 4",
            "mod_2: 5",
            "mult_3: 6",
        ]

        # 2. Modulo in loop filter 'when' clause and built-in mod() function
        text_when = """
@#for i in 1:12 when i % 4 == 0
div_by_4: @{i}, mod_3: @{i % 3}, mod_func: @{mod(i, 5)};
@#endfor
"""
        out_when = preprocess_macro(text_when).strip()
        lines_when = [l.strip() for l in out_when.splitlines() if l.strip()]
        assert len(lines_when) == 3
        assert lines_when[0] == "div_by_4: 4, mod_3: 1, mod_func: 4;"
        assert lines_when[1] == "div_by_4: 8, mod_3: 2, mod_func: 3;"
        assert lines_when[2] == "div_by_4: 12, mod_3: 0, mod_func: 2;"

        # 3. Modulo by zero must raise DynareMacroError
        with pytest.raises(DynareMacroError, match="Modulo by zero"):
            preprocess_macro("@#define M = 10 % 0")

    def test_string_concatenation_and_types(self):
        """Test string concatenation operator + across strings, numbers, and expressions."""
        text = """
@#define PREFIX = "var_"
@#for i in 1:3
@{PREFIX + i + "_suffix"} = @{i * 10};
@{PREFIX + i} = @{i};
@#endfor
"""
        out = preprocess_macro(text).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        assert lines == [
            "var_1_suffix = 10;",
            "var_1 = 1;",
            "var_2_suffix = 20;",
            "var_2 = 2;",
            "var_3_suffix = 30;",
            "var_3 = 3;",
        ]

    def test_dynamic_include_and_cycle_detection(self):
        """Test dynamically constructed include paths, relative path resolution, and cycle detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            # Create modular include files
            (tmp_path / "model_us.inc").write_text("param_us = 1.0;\n", encoding="utf-8")
            (tmp_path / "model_eu.inc").write_text("param_eu = 2.0;\n", encoding="utf-8")
            (tmp_path / "model_jp.inc").write_text("param_jp = 3.0;\n", encoding="utf-8")

            # Dynamic include inside loop
            text = """
@#for country in ["us", "eu", "jp"]
  @#define INC_NAME = "model_" + country + ".inc"
  @#include INC_NAME
@#endfor
"""
            out = preprocess_macro(text, base_dir=tmp_path).strip()
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            assert lines == [
                "param_us = 1.0;",
                "param_eu = 2.0;",
                "param_jp = 3.0;",
            ]

            # Test circular include detection
            (tmp_path / "circ_a.inc").write_text('@#include "circ_b.inc"\n', encoding="utf-8")
            (tmp_path / "circ_b.inc").write_text('@#include "circ_a.inc"\n', encoding="utf-8")
            with pytest.raises(DynareMacroError, match="circular @#include detected"):
                preprocess_macro('@#include "circ_a.inc"', base_dir=tmp_path)

            # Test missing include file
            with pytest.raises(DynareMacroError, match="not found"):
                preprocess_macro('@#include "nonexistent_file.inc"', base_dir=tmp_path)

    def test_cartesian_products_and_power(self):
        """Test Cartesian product operator *, 3-way products, and Cartesian power ^."""
        text = """
@#define PROD2 = [1, 2] * ["a", "b"]
@#for (i, j) in PROD2
pair_@{i}_@{j};
@#endfor

@#define PROD3 = [1, 2] * ["x", "y"] * [10, 20]
@#for (i, j, k) in PROD3
triplet_@{i}_@{j}_@{k};
@#endfor

@#define POW2 = ["u", "v"] ^ 2
@#for (a, b) in POW2
pow_@{a}_@{b};
@#endfor
"""
        out = preprocess_macro(text).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        # 4 pairs + 8 triplets + 4 power pairs = 16
        assert len(lines) == 16
        assert "pair_1_a;" in lines
        assert "pair_2_b;" in lines
        assert "triplet_1_x_10;" in lines
        assert "triplet_2_y_20;" in lines
        assert "pow_u_u;" in lines
        assert "pow_v_v;" in lines

    def test_destructuring_tuples_and_error_handling(self):
        """Test tuple unpacking in @#for loops and error recovery on element count mismatches."""
        # Valid 3-element unpack
        text = """
@#for (a, b, c) in [(1, 2, 3), (4, 5, 6)]
sum_@{a} = @{a + b + c};
@#endfor
"""
        out = preprocess_macro(text).strip()
        assert out == "sum_1 = 6;\nsum_4 = 15;"

        # Mismatched target count: 2 targets for 3 elements
        with pytest.raises(DynareMacroError, match="Cannot unpack 3 elements into 2 variables"):
            preprocess_macro("@#for (x, y) in [(1, 2, 3)]\nfoo\n@#endfor")

        # Unpack non-sequence
        with pytest.raises(DynareMacroError, match="Cannot unpack non-sequence"):
            preprocess_macro("@#for (x, y) in [42]\nfoo\n@#endfor")

    def test_error_recovery_and_syntax_diagnostics(self):
        """Test macro preprocessor diagnostics on unclosed blocks, unknown variables, and syntax errors."""
        # Unclosed @#for
        with pytest.raises(DynareMacroError, match="Unclosed @#for block"):
            preprocess_macro("@#for i in 1:5\nval_@{i}\n")

        # Unclosed @#if
        with pytest.raises(DynareMacroError, match="Unclosed @#if block"):
            preprocess_macro("@#if 1 > 0\nval = 1;\n")

        # Undefined variable evaluation
        with pytest.raises(DynareMacroError, match="Undefined macro variable 'UNDEFINED_VAR'"):
            preprocess_macro("var = @{UNDEFINED_VAR};")

        # Division by zero
        with pytest.raises(DynareMacroError, match="Division by zero"):
            preprocess_macro("@#define ERR = 100 / 0")

    def test_full_dynare_mod_integration_with_macro_and_modulo(self):
        """Verify load_mod transparently parses, preprocesses (including % modulo), and solves models."""
        mod_text = """
% Top-level model comment
var y pi a;
varexo eps_a eps_m;
parameters beta sigma phi_pi phi_y rho_a;
beta = 0.99;
sigma = 1.0;
phi_pi = 1.5;
phi_y = 0.1;
rho_a = 0.8;

@#for i in 1:4
  @#if i % 2 == 0
    % Even-index metadata comment @{i}
  @#endif
@#endfor

model;
  a = rho_a * a(-1) + eps_a;
  y = y(+1) - (1/sigma) * (phi_pi * pi + phi_y * y + eps_m - pi(+1)) + a;
  pi = beta * pi(+1) + 0.05 * y;
end;

steady_state_model;
  a = 0;
  y = 0;
  pi = 0;
end;
"""
        model = load_mod(mod_text)
        assert isinstance(model, LinearModel)
        assert set(model.variables) == {"y", "pi", "a"}
        assert model.is_determinate is True
        irf = model.irf("eps_m", horizon=10)
        assert len(irf) == 11
        assert not irf.isna().any().any()


# ==============================================================================
# 2. Interactive IRF Widgets Adversarial Stress Tests
# ==============================================================================

class TestInteractiveIRFWidgetsStress:
    """Adversarial stress testing of live interactive parameter slider widgets."""

    def test_rapid_parameter_slider_latency_50_updates(self):
        """Measure latency over 50 consecutive parameter adjustments, asserting mean latency < 15ms."""
        model = _build_nk_model()
        res = model.interactive_irf(
            parameters=["phi_pi", "sigma", "kappa"],
            shocks=["eps_r"],
            horizon=20,
        )

        callback_latencies: list[float] = []
        phi_values = np.linspace(1.1, 3.5, 50)

        for phi in phi_values:
            res.set_value("phi_pi", float(phi))
            callback_latencies.append(res.last_latency_ms)

        lat = np.array(callback_latencies)
        mean_lat = float(lat.mean())
        max_lat = float(lat.max())
        p95_lat = float(np.percentile(lat, 95))

        print(f"\n[STRESS TEST] NK 50 updates callback latency: mean={mean_lat:.3f}ms, max={max_lat:.3f}ms, p95={p95_lat:.3f}ms")
        assert mean_lat < 15.0, f"Mean callback latency {mean_lat:.3f}ms exceeds 15ms SLA threshold"
        assert max_lat < 30.0, f"Max callback latency {max_lat:.3f}ms exceeds 30ms ceiling"
        assert res.n_updates >= 50
        plt.close("all")

    def test_smets_wouters_07_50_updates_latency(self):
        """Verify 50 consecutive updates on large Smets-Wouters (2007) model maintain mean latency < 15ms."""
        mod_path = Path("puremacro/dsge/_references/sw07_pfeifer.mod")
        if not mod_path.is_file():
            pytest.skip("sw07_pfeifer.mod not found")

        model = load_mod(mod_path)
        res = model.interactive_irf(
            parameters=["crr", "crhopinf"],
            shocks=["ea"],
            horizon=20,
        )

        callback_latencies: list[float] = []
        crr_values = np.linspace(0.60, 0.95, 50)

        for crr in crr_values:
            res.set_value("crr", float(crr))
            callback_latencies.append(res.last_latency_ms)

        lat = np.array(callback_latencies)
        mean_lat = float(lat.mean())
        max_lat = float(lat.max())
        p95_lat = float(np.percentile(lat, 95))

        print(f"\n[STRESS TEST] SW07 50 updates callback latency: mean={mean_lat:.3f}ms, max={max_lat:.3f}ms, p95={p95_lat:.3f}ms")
        assert mean_lat < 15.0, f"SW07 mean callback latency {mean_lat:.3f}ms exceeds 15ms SLA threshold"
        assert res.n_updates >= 50
        plt.close("all")

    def test_dragging_parameters_across_determinacy_boundaries(self):
        """Stress-test dragging Taylor rule phi_pi across determinacy boundary and back.

        Trajectory: 1.5 -> 0.5 -> 0.1 -> 1.2 -> 2.0 -> 0.0 -> 1.5
        Asserts:
        - When phi_pi < 1: is_determinate is False, lines are set to NaN without crash
        - Status warning banner is displayed during indeterminacy
        - When phi_pi > 1: is_determinate is True, valid numerical lines restore cleanly
        - Status warning banner is cleared
        - Restored trajectory matches independent direct model solve to machine precision
        - Export methods (.summary, .to_frame, .to_markdown, etc.) execute without error in both regimes
        """
        model = _build_nk_model()
        res = model.interactive_irf(
            parameters=["phi_pi", "sigma", "kappa"],
            shocks=["eps_r"],
            horizon=20,
        )

        # 1. Baseline: determinate
        assert res.model.is_determinate is True
        y_baseline = res.lines[("y", "eps_r")].get_ydata().copy()
        assert not np.isnan(y_baseline).any()
        assert res.status_text.get_text() == ""

        # 2. Drag across boundary to indeterminacy (phi_pi = 0.5)
        res.set_value("phi_pi", 0.5)
        assert res.model.is_determinate is False
        y_indet = res.lines[("y", "eps_r")].get_ydata()
        assert np.isnan(y_indet).all()
        assert "Blanchard-Kahn" in res.status_text.get_text()

        # Check export safety during indeterminacy
        df_indet = res.to_frame()
        assert np.isnan(df_indet.to_numpy()).all()
        assert "Indeterminate" in res.summary()
        assert len(res.to_markdown()) > 0
        assert len(res.to_latex()) > 0
        assert len(res.to_typst()) > 0

        # 3. Drag further into indeterminacy (phi_pi = 0.1)
        res.set_value("phi_pi", 0.1)
        assert res.model.is_determinate is False
        assert np.isnan(res.lines[("y", "eps_r")].get_ydata()).all()

        # 4. Restore across boundary back to determinacy (phi_pi = 1.2)
        res.set_value("phi_pi", 1.2)
        assert res.model.is_determinate is True
        y_restored = res.lines[("y", "eps_r")].get_ydata()
        assert not np.isnan(y_restored).any()
        assert res.status_text.get_text() == ""

        # 5. Drag to high determinacy (phi_pi = 2.0)
        res.set_value("phi_pi", 2.0)
        assert res.model.is_determinate is True
        y_20 = res.lines[("y", "eps_r")].get_ydata()
        assert not np.isnan(y_20).any()

        # Verify numerical fidelity against independent fresh solve
        fresh_model = _build_nk_model(phi_pi=2.0)
        fresh_irf = fresh_model.irf("eps_r", horizon=20)
        np.testing.assert_allclose(y_20, fresh_irf["y"].to_numpy(), rtol=1e-10, atol=1e-10)

        # 6. Drag back to deep indeterminacy (phi_pi = 0.0)
        res.set_value("phi_pi", 0.0)
        assert res.model.is_determinate is False
        assert np.isnan(res.lines[("y", "eps_r")].get_ydata()).all()
        assert "Blanchard-Kahn" in res.status_text.get_text()

        # 7. Clean restoration back to baseline calibration (phi_pi = 1.5)
        res.set_value("phi_pi", 1.5)
        assert res.model.is_determinate is True
        assert res.status_text.get_text() == ""
        np.testing.assert_allclose(res.lines[("y", "eps_r")].get_ydata(), y_baseline, rtol=1e-10, atol=1e-10)

        plt.close("all")

    def test_multi_parameter_updates_and_reset(self):
        """Test simultaneous multi-parameter batch updates and reset button behavior."""
        model = _build_nk_model()
        res = model.interactive_irf(
            parameters=["phi_pi", "sigma", "kappa"],
            shocks=["eps_r"],
            horizon=20,
        )
        # Batch update
        res.update({"phi_pi": 2.2, "sigma": 1.5, "kappa": 0.08})
        vals = res.get_values()
        assert vals["phi_pi"] == pytest.approx(2.2)
        assert vals["sigma"] == pytest.approx(1.5)
        assert vals["kappa"] == pytest.approx(0.08)
        assert res.model.is_determinate is True

        # Reset back to baseline
        res.reset()
        reset_vals = res.get_values()
        assert reset_vals["phi_pi"] == pytest.approx(1.5)
        assert reset_vals["sigma"] == pytest.approx(1.0)
        assert reset_vals["kappa"] == pytest.approx(0.05)

        plt.close("all")

    def test_tuple_unpacking_and_cleanup(self):
        """Test tuple unpacking interface and disconnect cleanup."""
        model = _build_nk_model()
        fig, sliders = model.interactive_irf(
            parameters=["phi_pi", "kappa"],
            shocks=["eps_r"],
            horizon=10,
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(sliders, dict)
        assert len(sliders) == 2

        # Verify widget registry on figure
        widget_res = getattr(fig, "_interactive_widget", None)
        assert isinstance(widget_res, InteractiveIRFResult)
        widget_res.disconnect()
        for s in widget_res.sliders.values():
            assert not s.eventson

        plt.close("all")
