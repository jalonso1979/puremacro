"""Unit tests for puremacro DSGE model utilities and extensions (Milestone 4).

Tests:
1. model_info() schema, dynamic variable classifications, lead/lag incidence,
   Jacobian sparsity metrics, and presentation contract (.summary(), .to_markdown(), .to_latex(), .to_typst()).
2. detect_linear_model() and detect_linearity alias for linear and non-linear systems.
3. write_latex_dynamic_model() LaTeX equation exporter, dynamic subscripts,
   Greek parameter rendering, math formatting, and equation tags.
4. Parser boundary polish: unclosed model block, duplicate symbols, and local '#' collision.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from puremacro.dsge._ast import BinOp, Call, Const, Node, Param, Var
from puremacro.dsge._parser import DynareParseError, parse_mod_to_dag
from puremacro.dsge._utils import (
    ModelInfoResult,
    detect_linear_model,
    detect_linearity,
    model_info,
    write_latex_dynamic_model,
)
from puremacro.dsge.build import ModelError
from puremacro.dsge.dynare import build_dynare, load_mod

_LINEAR_RBC = """
var c k a;
varexo eps;
parameters alpha beta delta rho;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.95;
model(linear);
c = c(+1) - (1 - beta*(1-delta)) * a(+1) + alpha * (1 - beta*(1-delta)) * k;
k = (1-delta)*k(-1) + delta*a - delta*c;
a = rho*a(-1) + eps;
end;
"""

_NONLINEAR_RBC = """
var c k a;
varexo eps;
parameters alpha beta delta rho gamma;
alpha = 0.33; beta = 0.99; delta = 0.025; rho = 0.95; gamma = 2.0;
model;
c^(-gamma) = beta * c(+1)^(-gamma) * (alpha * exp(a(+1)) * k^(alpha-1) + 1 - delta);
k = exp(a) * k(-1)^alpha - c + (1-delta)*k(-1);
a = rho * a(-1) + eps;
end;
initval;
k = 38.0; c = 2.0; a = 0.0;
end;
"""


# ===========================================================================
# 1. model_info() Tests
# ===========================================================================


class TestModelInfo:
    """Test model_info() output schema, classifications, incidence, and presentation."""

    def test_model_info_basic_counts(self):
        """Verify equation, variable, shock, and parameter counts."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        info = model_info(dag)

        assert isinstance(info, ModelInfoResult)
        assert info.num_equations == 3
        assert info.num_variables == 3
        assert info.num_shocks == 1
        assert info.num_parameters == 4
        assert info.variable_names == ("c", "k", "a")
        assert info.shock_names == ("eps",)
        assert set(info.parameter_names) == {"alpha", "beta", "delta", "rho"}
        assert info.is_linear is True

    def test_model_info_dynamic_classification(self):
        """Verify dynamic variable classification into static, forward, backward, mixed."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        info = model_info(dag)

        # In RBC:
        # c appears at t, t+1 -> purely forward
        # k appears at t-1, t -> purely backward
        # a appears at t-1, t, t+1 -> mixed
        assert "c" in info.purely_forward
        assert "k" in info.purely_backward
        assert "a" in info.mixed
        assert len(info.purely_static) == 0

    def test_model_info_states_and_controls(self):
        """Verify predetermined states and forward controls classification."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        info = model_info(dag)

        assert set(info.states) == {"k", "a"}
        assert info.controls == ("c",)

    def test_model_info_purely_static_model(self):
        """Verify model with only contemporaneous variables classifies all as purely static."""
        src = """
        var x y;
        varexo e;
        parameters a;
        a = 1.0;
        model;
        x = y + e;
        y = a * e;
        end;
        """
        dag = parse_mod_to_dag(src)
        info = model_info(dag)

        assert set(info.purely_static) == {"x", "y"}
        assert len(info.purely_forward) == 0
        assert len(info.purely_backward) == 0
        assert len(info.mixed) == 0
        assert len(info.states) == 0
        assert set(info.controls) == {"x", "y"}

    def test_model_info_lead_lag_incidence_structure(self):
        """Verify lead_lag_incidence matrix dimensions and sequential numbering."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        info = model_info(dag)

        mat = info.lead_lag_incidence
        assert isinstance(mat, np.ndarray)
        assert mat.shape == (3, 3)
        assert mat.dtype in (np.int64, np.int32, int)

        # Variables: c (col 0), k (col 1), a (col 2)
        # Row 0 (t-1): k and a appear -> non-zero
        assert mat[0, 0] == 0  # c(-1) does not appear
        assert mat[0, 1] > 0   # k(-1) appears
        assert mat[0, 2] > 0   # a(-1) appears

        # Row 1 (t): c, k, a all appear
        assert mat[1, 0] > 0
        assert mat[1, 1] > 0
        assert mat[1, 2] > 0

        # Row 2 (t+1): c(+1) and a(+1) appear
        assert mat[2, 0] > 0   # c(+1) appears
        assert mat[2, 1] == 0  # k(+1) does not appear
        assert mat[2, 2] > 0   # a(+1) appears

    def test_model_info_jacobian_sparsity(self):
        """Verify Jacobian sparsity density floats."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        info = model_info(dag)

        sparsity = info.jacobian_sparsity
        assert "A_plus" in sparsity
        assert "A_0" in sparsity
        assert "A_minus" in sparsity
        assert "B_u" in sparsity

        assert isinstance(sparsity["A_plus"], float)
        assert isinstance(sparsity["A_0"], float)
        assert isinstance(sparsity["A_minus"], float)
        assert isinstance(sparsity["B_u"], float)

        # 3 equations, 3 variables -> 9 entries per endogenous block
        # A_plus has c(+1), a(+1) in eq 1 -> 2 entries -> 2/9
        assert sparsity["A_plus"] == pytest.approx(2.0 / 9.0)
        # A_0 has c, k in eq 1; k, a, c in eq 2; a in eq 3 -> 6 entries -> 6/9
        assert sparsity["A_0"] == pytest.approx(6.0 / 9.0)
        # A_minus has k(-1) in eq 2; a(-1) in eq 3 -> 2 entries -> 2/9
        assert sparsity["A_minus"] == pytest.approx(2.0 / 9.0)
        # B_u has eps in eq 3 -> 1 entry out of 3*1 -> 1/3
        assert sparsity["B_u"] == pytest.approx(1.0 / 3.0)

    def test_model_info_presentation_contract(self):
        """Verify summary, to_frame, to_markdown, to_latex, to_typst methods."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        info = model_info(dag)

        # DataFrame
        df = info.to_frame()
        assert isinstance(df, pd.DataFrame)
        assert "Number of equations" in df.index
        assert "Number of variables" in df.index

        # Summary
        summary = info.summary()
        assert "Dynare Model Information" in summary
        assert "Equations" in summary
        assert "Variables" in summary
        assert "Lead-Lag Incidence Matrix" in summary

        # Markdown
        md = info.to_markdown()
        assert "|" in md
        assert "Number of equations" in md

        # LaTeX
        tex = info.to_latex()
        assert "tabular" in tex
        assert "Number of equations" in tex

        # Typst
        typ = info.to_typst()
        assert "table" in typ

    def test_model_info_immutability(self):
        """Verify ModelInfoResult is frozen and attributes cannot be mutated."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        info = model_info(dag)

        with pytest.raises((AttributeError, TypeError)):
            info.num_equations = 100

    def test_model_info_on_linear_model_instance(self):
        """Verify model_info works seamlessly on solved LinearModel instances."""
        def toy_eqs(lead, curr, lag, e, p):
            return [curr.c - p.rho * lag.c - e.eps]

        m = build_dynare(
            toy_eqs,
            variables=["c"],
            shocks=["eps"],
            params={"rho": 0.8},
            guess={"c": 0.0},
        )
        info = model_info(m)
        assert info.num_equations == 1
        assert info.num_variables == 1
        assert info.num_shocks == 1
        assert info.is_linear is True
        assert "c" in info.states


# ===========================================================================
# 2. Linearity Detection Tests
# ===========================================================================


class TestLinearityDetection:
    """Test detect_linear_model() and detect_linearity alias."""

    def test_explicit_linear_flag(self):
        """Explicit model(linear); is detected as linear."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        assert detect_linear_model(dag) is True
        assert detect_linearity(dag) is True

    def test_implicit_linear_model(self):
        """Linear equations without explicit flag are detected as linear."""
        src = """
        var c k;
        varexo e;
        parameters a;
        a = 0.9;
        model;
        c = a * c(-1) + e;
        k = 0.5 * k(-1) + c;
        end;
        """
        dag = parse_mod_to_dag(src)
        assert detect_linear_model(dag) is True
        assert detect_linearity(dag) is True

    def test_affine_model_with_constant_offsets(self):
        """Affine model with constant level offsets is detected as linear."""
        src = """
        var y;
        varexo e;
        parameters a b;
        a = 0.8; b = 1.5;
        model;
        y = b + a * y(-1) + e;
        end;
        """
        dag = parse_mod_to_dag(src)
        assert detect_linear_model(dag) is True

    def test_steady_state_operator_linear(self):
        """Equations with STEADY_STATE() operator remain linear."""
        src = """
        var c;
        varexo e;
        parameters rho;
        rho = 0.8;
        model;
        c - STEADY_STATE(c) = rho * (c(-1) - STEADY_STATE(c)) + e;
        end;
        """
        dag = parse_mod_to_dag(src)
        assert detect_linear_model(dag) is True

    def test_diff_operator_linear(self):
        """Equations with diff() operator remain linear."""
        src = """
        var y dy;
        varexo e;
        parameters rho;
        rho = 0.8;
        model;
        dy = diff(y);
        y = rho * y(-1) + e;
        end;
        """
        dag = parse_mod_to_dag(src)
        assert detect_linear_model(dag) is True

    def test_power_term_nonlinear(self):
        """Power terms c^gamma detected as non-linear."""
        dag = parse_mod_to_dag(_NONLINEAR_RBC)
        assert detect_linear_model(dag) is False
        assert detect_linearity(dag) is False

    def test_bilinear_interaction_nonlinear(self):
        """Bilinear interactions x * y detected as non-linear."""
        src = """
        var x y;
        varexo e;
        model;
        x = x(-1) * y(-1) + e;
        y = e;
        end;
        """
        dag = parse_mod_to_dag(src)
        assert detect_linear_model(dag) is False

    def test_transcendental_function_nonlinear(self):
        """Transcendental functions (exp, log) detected as non-linear."""
        src = """
        var y;
        varexo e;
        model;
        y = exp(y(-1)) + e;
        end;
        """
        dag = parse_mod_to_dag(src)
        assert detect_linear_model(dag) is False


# ===========================================================================
# 3. write_latex_dynamic_model() Tests
# ===========================================================================


class TestLaTeXExport:
    """Test publication-grade LaTeX dynamic model rendering."""

    def test_latex_export_align_block(self):
        """LaTeX export wraps equations in \\begin{align} and \\end{align}."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        tex = write_latex_dynamic_model(dag)

        assert "\\begin{align" in tex
        assert "\\end{align" in tex
        assert tex.count("\\begin{align") == tex.count("\\end{align")

    def test_latex_export_dynamic_subscripts(self):
        """Dynamic subscripts render t+1, t, t-1."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        tex = write_latex_dynamic_model(dag)

        assert "t+1" in tex
        assert "t-1" in tex
        assert "t}" in tex

    def test_latex_export_greek_parameters(self):
        """Greek parameter names (alpha, beta, delta, rho) render with backslashes."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        tex = write_latex_dynamic_model(dag)

        assert r"\alpha" in tex
        assert r"\beta" in tex
        assert r"\delta" in tex
        assert r"\rho" in tex

    def test_latex_export_compound_parameters(self):
        """Compound parameter names like phi_pi render as \\phi_{\\pi}."""
        src = """
        var r pi;
        varexo e;
        parameters phi_pi;
        phi_pi = 1.5;
        model;
        r = phi_pi * pi + e;
        pi = 0.5 * pi(+1) + r;
        end;
        """
        dag = parse_mod_to_dag(src)
        tex = write_latex_dynamic_model(dag)

        assert r"\phi" in tex

    def test_latex_export_math_functions(self):
        """Math functions (exp, log, sqrt) render with backslashes."""
        src = """
        var y;
        varexo e;
        model;
        y = exp(log(y(-1))) + sqrt(e);
        end;
        """
        dag = parse_mod_to_dag(src)
        tex = write_latex_dynamic_model(dag)

        assert r"\exp" in tex
        assert r"\ln" in tex or r"\log" in tex
        assert r"\sqrt" in tex

    def test_latex_export_equation_tags(self):
        """Equation tags [name='Euler'] render with \\tag{Euler}."""
        src = """
        var c;
        varexo e;
        parameters a;
        a = 1.0;
        model;
        [name='Euler'] c = a * c(-1) + e;
        end;
        """
        dag = parse_mod_to_dag(src)
        tex_with_tags = write_latex_dynamic_model(dag, write_equation_tags=True)
        assert r"\tag{Euler}" in tex_with_tags

        tex_without_tags = write_latex_dynamic_model(dag, write_equation_tags=False)
        assert r"\tag" not in tex_without_tags

    def test_latex_export_to_file(self, tmp_path):
        """Writing LaTeX output to filepath creates valid UTF-8 file."""
        dag = parse_mod_to_dag(_LINEAR_RBC)
        out_path = tmp_path / "subdir" / "rbc_model.tex"
        tex = write_latex_dynamic_model(dag, filepath=out_path)

        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert content == tex
        assert "\\begin{align" in content


# ===========================================================================
# 4. Parser Boundary Polish Verification Tests
# ===========================================================================


class TestParserBoundaryPolish:
    """Verify the 3 boundary items escalated from TEST_READY.md § 5."""

    def test_unclosed_model_block_raises_dynare_parse_error(self):
        """Unclosed model block reaching EOF raises DynareParseError."""
        src = "var c; varexo e; parameters a; a=1; model; c = e;"
        with pytest.raises(DynareParseError, match="Unclosed model block"):
            parse_mod_to_dag(src)

    def test_duplicate_variable_declaration_raises_dynare_parse_error(self):
        """Declaring the same variable twice raises DynareParseError."""
        src_same_block = "var c c; varexo e; parameters a; a=1; model; c = e; end;"
        with pytest.raises(DynareParseError, match="declared twice"):
            parse_mod_to_dag(src_same_block)

        src_cross_block = "var c; parameters c; model; c = 1; end;"
        with pytest.raises(DynareParseError, match="declared twice"):
            parse_mod_to_dag(src_cross_block)

    def test_local_hash_variable_colliding_with_endogenous_raises_model_error(self):
        """Model-local '#' variable colliding with endogenous variable raises ModelError."""
        src = "var c; varexo e; parameters a; a=1; model; #c = 2.0; c = e; end;"
        with pytest.raises(ModelError, match="collides with declared endogenous variable"):
            parse_mod_to_dag(src)
