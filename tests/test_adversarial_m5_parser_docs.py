"""Adversarial challenge tests for puremacro 2.9.0 Tier 3 M5 deliverables.

Covers:
1. Mixed shocks block:
   - Partial variance declarations trigger UserWarning naming ONLY undeclared shocks.
   - Fully declared shocks trigger NO UserWarning.
   - Omitted shocks block triggers NO UserWarning (defaults to identity matrix).
   - Covariance matrix numerical structure matches declarations.
   - Parser bug detection: 'var eps; variance EXPR;' after SEMI is not recognized.
2. Steady state model evaluation:
   - Complex chained expressions evaluating earlier temporary / endogenous definitions.
   - Interleaved parameter dependencies.
   - Failure modes: division by zero, undefined names raise ValueError with context.
3. Perturbation orders:
   - Order 1, 2, 3 succeed and return respective model / solution types.
   - Order 4, 0, negative orders raise ValueError with informative message.
4. Documentation code block execution and compilation verification for
   docs/dsge_parity_surface.md and docs/es/dsge_parity_surface.md.
"""
import math
import re
import warnings
from pathlib import Path

import numpy as np
import pytest

from puremacro.dsge.dynare import parse_mod, load_mod
from puremacro.dsge.build import LinearModel
from puremacro.dsge.pruning import PrunedDSGESolution, Order3PrunedSolution
from puremacro.dsge.extended_path import extended_path
from puremacro.dsge.conditional import conditional_forecast
from puremacro.dsge.parity import ParityDashboardResult


RBC_MOD_TEXT = """
var y c k a;
varexo eps_a;
parameters alpha beta delta rho;
alpha = 0.33;
beta = 0.99;
delta = 0.025;
rho = 0.95;

model;
1/c = beta * (1/c(+1)) * (alpha * exp(a(+1)) * k^(alpha - 1) + 1 - delta);
c + k = exp(a) * k(-1)^alpha + (1 - delta) * k(-1);
y = exp(a) * k(-1)^alpha;
a = rho * a(-1) + eps_a;
end;

initval;
k = 10.0;
c = 0.8;
y = 1.0;
a = 0.0;
end;

steady_state_model;
a = 0.0;
k = ((1/beta - 1 + delta) / alpha)^(1 / (alpha - 1));
y = k^alpha;
c = y - delta * k;
end;

shocks;
var eps_a; stderr 0.01;
end;
"""


# =============================================================================
# 1. Mixed Shocks Block Warning Stress Tests
# =============================================================================

class TestMixedShocksWarning:
    """Stress test UserWarning behavior when some shocks have variances and some do not."""

    def test_mixed_shocks_single_undeclared(self):
        """When 2 of 3 shocks are declared, UserWarning fires ONLY for the 1 undeclared shock."""
        mod = """
        var y, c, k;
        varexo eps_a, eps_b, eps_c;
        parameters alpha, beta;
        alpha = 0.33;
        beta = 0.99;

        model;
        y = alpha * k(-1) + eps_a + eps_b + eps_c;
        c = y;
        k = 0.5 * k(-1) + 0.5 * y;
        end;

        shocks;
        var eps_a; stderr 0.01;
        var eps_b = 0.04;
        end;
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = parse_mod(mod)

        # Check UserWarning fired
        user_warnings = [item for item in w if issubclass(item.category, UserWarning)]
        assert len(user_warnings) == 1, f"Expected 1 UserWarning, got {len(user_warnings)}"

        msg = str(user_warnings[0].message)
        assert "parse_mod: the shocks; block declares no variance for" in msg
        assert "['eps_c']" in msg
        # eps_a and eps_b should NOT be in the undeclared list
        assert "eps_a" not in msg
        assert "eps_b" not in msg

        # Numerical verification of covariance matrix
        cov = parsed["shock_cov"]
        assert cov.shape == (3, 3)
        assert parsed["shocks"] == ["eps_a", "eps_b", "eps_c"]
        np.testing.assert_allclose(cov[0, 0], 0.01**2, rtol=1e-12)
        np.testing.assert_allclose(cov[1, 1], 0.04, rtol=1e-12)
        assert cov[2, 2] == 0.0
        assert np.all(cov[2, :] == 0.0)
        assert np.all(cov[:, 2] == 0.0)

    def test_mixed_shocks_multiple_undeclared(self):
        """Multiple undeclared shocks are accurately listed in UserWarning."""
        mod = """
        var y;
        varexo e1, e2, e3, e4;
        parameters rho;
        rho = 0.8;
        model;
        y = rho * y(-1) + e1 + e2 + e3 + e4;
        end;
        shocks;
        var e2; stderr 0.05;
        end;
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = parse_mod(mod)

        user_warnings = [item for item in w if issubclass(item.category, UserWarning)]
        assert len(user_warnings) == 1
        msg = str(user_warnings[0].message)
        assert "['e1', 'e3', 'e4']" in msg
        assert "e2" not in msg

        cov = parsed["shock_cov"]
        assert cov.shape == (4, 4)
        np.testing.assert_allclose(cov[1, 1], 0.05**2, rtol=1e-12)
        assert cov[0, 0] == 0.0
        assert cov[2, 2] == 0.0
        assert cov[3, 3] == 0.0

    def test_all_shocks_declared_no_warning(self):
        """When all shocks have variance/stderr declared, no UserWarning is raised."""
        mod = """
        var y;
        varexo e1, e2;
        parameters rho;
        rho = 0.8;
        model;
        y = rho * y(-1) + e1 + e2;
        end;
        shocks;
        var e1; stderr 0.01;
        var e2 = 0.02;
        end;
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = parse_mod(mod)

        user_warnings = [item for item in w if issubclass(item.category, UserWarning)]
        assert len(user_warnings) == 0, f"Unexpected warnings: {[str(x.message) for x in user_warnings]}"
        cov = parsed["shock_cov"]
        np.testing.assert_allclose(cov[0, 0], 0.0001, rtol=1e-12)
        np.testing.assert_allclose(cov[1, 1], 0.02, rtol=1e-12)

    def test_no_shocks_block_no_warning_identity_cov(self):
        """When no shocks; block exists, shock_cov is identity and no warning is raised."""
        mod = """
        var y;
        varexo e1, e2;
        parameters rho;
        rho = 0.8;
        model;
        y = rho * y(-1) + e1 + e2;
        end;
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = parse_mod(mod)

        user_warnings = [item for item in w if issubclass(item.category, UserWarning)]
        assert len(user_warnings) == 0
        cov = parsed["shock_cov"]
        np.testing.assert_allclose(cov, np.eye(2))

    def test_variance_keyword_syntax_supported(self):
        """Dynare syntax 'var e2; variance 0.04;' is parsed after SEMI,
        storing the variance in shocks_config, setting the covariance diagonal,
        and emitting 0 false-positive warnings."""
        mod = """
        var y;
        varexo e1, e2;
        parameters rho;
        rho = 0.8;
        model;
        y = rho * y(-1) + e1 + e2;
        end;
        shocks;
        var e1; stderr 0.01;
        var e2; variance 0.04;
        end;
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = parse_mod(mod)

        user_warnings = [item for item in w if issubclass(item.category, UserWarning)]
        assert len(user_warnings) == 0, f"Expected 0 warnings, got {[str(x.message) for x in user_warnings]}"
        assert "e2" in parsed["_dag"].shocks_config["variances"]
        cov = parsed["shock_cov"]
        np.testing.assert_allclose(cov[0, 0], 0.01**2, rtol=1e-12)
        np.testing.assert_allclose(cov[1, 1], 0.04, rtol=1e-12)

    def test_mixed_shocks_syntax_stderr_variance_assign(self):
        """Mixed shocks; block with var e1; stderr 0.01; var e2; variance 0.04; var e3 = 0.09;.
        Asserts:
        - shock_cov has correct diagonals (0.01^2, 0.04, 0.09).
        - 0 false-positive UserWarning are emitted.
        """
        mod = """
        var y, c, k;
        varexo e1, e2, e3;
        parameters rho;
        rho = 0.8;
        model;
        y = rho * y(-1) + e1 + e2 + e3;
        c = 0.8 * y;
        k = 0.5 * k(-1) + 0.5 * y;
        end;
        shocks;
        var e1; stderr 0.01;
        var e2; variance 0.04;
        var e3 = 0.09;
        end;
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = parse_mod(mod)

        user_warnings = [item for item in w if issubclass(item.category, UserWarning)]
        assert len(user_warnings) == 0, f"Expected 0 warnings, got {[str(x.message) for x in user_warnings]}"

        cov = parsed["shock_cov"]
        assert cov.shape == (3, 3)
        np.testing.assert_allclose(cov[0, 0], 0.01**2, rtol=1e-12)
        np.testing.assert_allclose(cov[1, 1], 0.04, rtol=1e-12)
        np.testing.assert_allclose(cov[2, 2], 0.09, rtol=1e-12)
        assert cov[0, 1] == 0.0
        assert cov[0, 2] == 0.0
        assert cov[1, 2] == 0.0

    def test_shocks_scientific_notation_expressions_comments_whitespace(self):
        """Adversarial test with scientific notation, expressions (0.02^2), whitespace, inline comments."""
        mod = """
        var y, a, b;
        varexo e1, e2, e3, e4, e5;
        parameters alpha, sig_base;
        alpha = 0.33;
        sig_base = 0.01;
        model;
        y = alpha * y(-1) + e1 + e2 + e3 + e4 + e5;
        a = 0.9 * a(-1) + e1;
        b = 0.9 * b(-1) + e2;
        end;
        shocks;
          // Line comment with //
          var   e1   ;  
          stderr   1e-3  ;   /* scientific notation */

          var e2; variance 0.02^2; // expression with exponentiation

          var e3 = (0.03 * 2)^2; % inline comment with %

          /* Multi-line
             block comment
          */
          var e4; stderr sig_base * 2; // expression with parameter

          var   e5   ;
          variance   
          2.5e-4 ; // multiline whitespace and scientific notation
        end;
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            parsed = parse_mod(mod)

        user_warnings = [item for item in w if issubclass(item.category, UserWarning)]
        assert len(user_warnings) == 0, f"Expected 0 warnings, got {[str(x.message) for x in user_warnings]}"

        cov = parsed["shock_cov"]
        assert cov.shape == (5, 5)
        # e1: stderr 1e-3 -> var is (1e-3)^2 = 1e-6
        np.testing.assert_allclose(cov[0, 0], (1e-3)**2, rtol=1e-12)
        # e2: variance 0.02^2 -> var is 0.0004
        np.testing.assert_allclose(cov[1, 1], 0.02**2, rtol=1e-12)
        # e3: var = (0.03 * 2)^2 = 0.06^2 = 0.0036
        np.testing.assert_allclose(cov[2, 2], 0.06**2, rtol=1e-12)
        # e4: stderr 0.01 * 2 = 0.02 -> var is 0.02^2 = 0.0004
        np.testing.assert_allclose(cov[3, 3], (0.01 * 2)**2, rtol=1e-12)
        # e5: variance 2.5e-4
        np.testing.assert_allclose(cov[4, 4], 2.5e-4, rtol=1e-12)



# =============================================================================
# 2. Steady State Model Chained Expressions & Scopes Stress Tests
# =============================================================================

class TestSteadyStateModelScope:
    """Stress test chained evaluation of local variables in steady_state_model."""

    def test_chained_local_variables_in_steady_state(self):
        """Variables defined in steady_state_model can be referenced by subsequent lines."""
        mod = """
        var y, c, k, r, w;
        varexo eps;
        parameters alpha, beta, delta, A;
        alpha = 0.33;
        beta = 0.99;
        delta = 0.025;
        A = 1.0;

        model;
        c = y - delta * k;
        y = A * k(-1)^alpha;
        r = alpha * A * k(-1)^(alpha - 1);
        w = (1 - alpha) * A * k(-1)^alpha;
        k = 0.9 * k(-1) + 0.1 * y + eps;
        end;

        steady_state_model;
        // Step 1: define intermediate interest rate
        r_target = 1 / beta - 1 + delta;
        // Step 2: use r_target to compute capital k
        k = (r_target / (alpha * A))^(1 / (alpha - 1));
        // Step 3: use k to compute output y
        y = A * k^alpha;
        // Step 4: use y and k to compute consumption c
        c = y - delta * k;
        // Step 5: use y and k to compute wage w
        w = (1 - alpha) * y;
        // Step 6: check r matches r_target
        r = r_target;
        end;
        """
        parsed = parse_mod(mod)
        ss = parsed["steady_state"]

        expected_r = 1 / 0.99 - 1 + 0.025
        expected_k = (expected_r / 0.33)**(1 / (0.33 - 1))
        expected_y = expected_k**0.33
        expected_c = expected_y - 0.025 * expected_k
        expected_w = (1 - 0.33) * expected_y

        np.testing.assert_allclose(ss["r"], expected_r, rtol=1e-10)
        np.testing.assert_allclose(ss["k"], expected_k, rtol=1e-10)
        np.testing.assert_allclose(ss["y"], expected_y, rtol=1e-10)
        np.testing.assert_allclose(ss["c"], expected_c, rtol=1e-10)
        np.testing.assert_allclose(ss["w"], expected_w, rtol=1e-10)

    def test_steady_state_division_by_zero_raises_value_error(self):
        """Division by zero in steady_state_model raises ValueError with clear context."""
        mod = """
        var y;
        varexo eps;
        parameters a;
        a = 0;
        model;
        y = y(-1) + eps;
        end;
        steady_state_model;
        y = 10 / a;
        end;
        """
        with pytest.raises(ValueError, match="could not evaluate 'y' in the steady_state_model block"):
            parse_mod(mod)

    def test_steady_state_undefined_name_raises_value_error(self):
        """Referencing an undeclared identifier in steady_state_model raises ValueError."""
        mod = """
        var y;
        varexo eps;
        parameters alpha;
        alpha = 0.5;
        model;
        y = y(-1) + eps;
        end;
        steady_state_model;
        y = nonexistent_variable * 2;
        end;
        """
        with pytest.raises(ValueError, match="could not evaluate 'y' in the steady_state_model block"):
            parse_mod(mod)


# =============================================================================
# 3. Perturbation Orders Verification (1, 2, 3 vs. Invalid)
# =============================================================================

class TestPerturbationOrders:
    """Stress test perturbation order acceptance and rejection in load_mod."""

    def test_order_1_returns_linear_model(self):
        """order=1 returns a solved LinearModel."""
        m1 = load_mod(RBC_MOD_TEXT, order=1)
        assert isinstance(m1, LinearModel)
        dr = m1.decision_rules()
        assert dr is not None
        assert dr["ghx"] is not None
        assert dr["ghu"] is not None
        assert hasattr(dr, "ghx")
        assert hasattr(dr, "ghu")
        assert hasattr(m1, "theoretical_moments")

    def test_order_2_returns_pruned_solution(self):
        """order=2 returns a PrunedDSGESolution (Kim et al. 2008)."""
        m2 = load_mod(RBC_MOD_TEXT, order=2)
        assert isinstance(m2, PrunedDSGESolution)
        assert hasattr(m2, "girf")
        assert hasattr(m2, "stochastic_steady_state")
        dr = m2.decision_rules()
        assert dr["ghxx"] is not None
        assert dr["ghs2"] is not None
        assert hasattr(dr, "ghxx")
        assert hasattr(dr, "ghs2")

    def test_order_3_returns_third_order_solution(self):
        """order=3 returns an Order3PrunedSolution (Andreasen et al. 2018)."""
        m3 = load_mod(RBC_MOD_TEXT, order=3)
        assert isinstance(m3, Order3PrunedSolution)
        dr = m3.decision_rules()
        assert dr["ghxxx"] is not None
        assert dr["ghxss"] is not None
        assert hasattr(dr, "ghxxx")
        assert hasattr(dr, "ghxss")

    def test_decision_rules_contains_and_key_guard(self):
        """DynareDR, Dynare2ndDR, and Dynare3rdDR implement __contains__ and key guard in __getitem__.
        Asserts:
        - String existing keys return True on 'key in dr'.
        - String nonexistent keys return False on 'key in dr'.
        - Non-string keys (0, None, [1], 3.14) return False on 'key in dr' without TypeError.
        - dr[key] works for existing strings and raises KeyError for nonexistent or non-string keys.
        """
        m1 = load_mod(RBC_MOD_TEXT, order=1)
        dr1 = m1.decision_rules()
        m2 = load_mod(RBC_MOD_TEXT, order=2)
        dr2 = m2.decision_rules()
        m3 = load_mod(RBC_MOD_TEXT, order=3)
        dr3 = m3.decision_rules()

        for dr, expected_attr in [
            (dr1, "ghx"),
            (dr2, "ghxx"),
            (dr3, "ghxxx"),
        ]:
            # Valid string keys
            assert expected_attr in dr
            assert "ghu" in dr
            assert "ys" in dr
            assert dr[expected_attr] is not None
            assert dr["ghu"] is not None
            assert dr["ys"] is not None

            # Nonexistent string keys
            assert "nonexistent" not in dr
            assert "__dummy_field__" not in dr
            with pytest.raises(KeyError):
                _ = dr["nonexistent"]
            with pytest.raises(KeyError):
                _ = dr["__dummy_field__"]

            # 'extra' handling: by default not in dr unless dynamically attached
            assert "extra" not in dr
            with pytest.raises(KeyError):
                _ = dr["extra"]

            # Test non-string keys for __contains__: must return False, never TypeError
            non_string_keys = [
                0, 1, -1,           # Integers
                3.14, -0.01,        # Floats
                None,               # NoneType
                object(),           # Arbitrary object
                [], [1, 2],         # Lists
                {}, {"a": 1},       # Dicts
                (1,),               # Tuples
                set(), {1, 2},      # Sets
            ]
            for bad_key in non_string_keys:
                assert bad_key not in dr, f"{bad_key!r} unexpectedly matched in {type(dr).__name__}"

            # Test non-string keys for __getitem__: must raise KeyError, NEVER TypeError
            for bad_key in non_string_keys:
                try:
                    _ = dr[bad_key]
                    pytest.fail(f"{type(dr).__name__}[{bad_key!r}] did not raise KeyError")
                except KeyError:
                    pass  # Expected behavior
                except TypeError as e:
                    pytest.fail(f"{type(dr).__name__}[{bad_key!r}] raised TypeError instead of KeyError: {e}")

        # Verify dynamic 'extra' attribute support in __contains__ and __getitem__
        for dr in (dr1, dr2, dr3):
            object.__setattr__(dr, "extra", {"custom_info": 42})
            assert "extra" in dr
            assert "custom_info" in dr
            assert dr["extra"] == {"custom_info": 42}
            # Clean up dynamic attribute to leave instance clean
            object.__delattr__(dr, "extra")
            assert "extra" not in dr

    @pytest.mark.parametrize("invalid_order", [4, 5, 0, -1, -2, 10])
    def test_invalid_orders_raise_value_error(self, invalid_order):
        """Orders outside {1, 2, 3} raise ValueError specifying valid orders."""
        with pytest.raises(ValueError, match="unsupported perturbation order"):
            load_mod(RBC_MOD_TEXT, order=invalid_order)


# =============================================================================
# 4. Docs Code Blocks Execution & Compilation Stress Tests
# =============================================================================

class TestDocsCodeBlocksParitySurface:
    """Stress test syntax compilation and execution safety of dsge_parity_surface.md."""

    @pytest.fixture
    def en_doc_text(self):
        p = Path("docs/dsge_parity_surface.md")
        assert p.is_file(), f"{p} not found"
        return p.read_text(encoding="utf-8")

    @pytest.fixture
    def es_doc_text(self):
        p = Path("docs/es/dsge_parity_surface.md")
        assert p.is_file(), f"{p} not found"
        return p.read_text(encoding="utf-8")

    def test_en_and_es_code_blocks_compile(self, en_doc_text, es_doc_text):
        """All python code blocks in English and Spanish parity surface docs compile cleanly."""
        fence = re.compile(r"```python\n(.*?)```", re.S)
        en_blocks = fence.findall(en_doc_text)
        es_blocks = fence.findall(es_doc_text)

        assert len(en_blocks) == 5, f"Expected 5 python blocks in EN doc, got {len(en_blocks)}"
        assert len(es_blocks) == 5, f"Expected 5 python blocks in ES doc, got {len(es_blocks)}"

        for i, block in enumerate(en_blocks):
            try:
                compile(block, f"docs/dsge_parity_surface.md:block_{i}", "exec")
            except SyntaxError as e:
                pytest.fail(f"EN doc block {i} failed syntax compilation: {e}")

        for i, block in enumerate(es_blocks):
            try:
                compile(block, f"docs/es/dsge_parity_surface.md:block_{i}", "exec")
            except SyntaxError as e:
                pytest.fail(f"ES doc block {i} failed syntax compilation: {e}")

    def test_doc_code_blocks_api_discrepancies(self):
        """Empirically test documented API calls against real dsge classes.
        
        Verifies:
        1. StochSimulResult provides .simulated_moments (not .empirical_moments)
        2. ExtendedPathResult provides .to_frame() (not .to_dataframe())
        3. ConditionalForecastResult provides .shocks / .shock_paths() (not .implied_shocks)
        4. ParityDashboardResult provides .passed (not .all_passed)
        """
        # 1. StochSimulResult
        m = load_mod(RBC_MOD_TEXT)
        res_mc = m.stoch_simul(periods=20, simul_replic=5)
        assert hasattr(res_mc, "simulated_moments")
        assert res_mc.simulated_moments is not None
        assert not hasattr(res_mc, "empirical_moments")

        # 2. ExtendedPathResult
        ep_res = extended_path(model_or_equations=m, periods=10, horizon=10, seed=1)
        assert hasattr(ep_res, "to_frame")
        df_frame = ep_res.to_frame()
        assert df_frame is not None
        assert not hasattr(ep_res, "to_dataframe")

        # 3. ConditionalForecastResult
        cf_res = conditional_forecast(model=m, conditions={"y": [1.0, 1.0]}, horizon=2)
        assert hasattr(cf_res, "shocks")
        assert hasattr(cf_res, "shock_paths")
        assert cf_res.shocks is not None
        assert not hasattr(cf_res, "implied_shocks")

        # 4. ParityDashboardResult
        p_res = ParityDashboardResult()
        assert hasattr(p_res, "passed")
        assert isinstance(p_res.passed, bool)
        assert not hasattr(p_res, "all_passed")

    def test_docs_contain_correct_apis_and_no_obsolete_attributes(self, en_doc_text, es_doc_text):
        """English and Spanish docs contain the updated APIs and no obsolete attributes."""
        for doc_name, text in [("EN", en_doc_text), ("ES", es_doc_text)]:
            # Correct updated APIs present
            assert "simulated_moments" in text, f"{doc_name} missing simulated_moments"
            assert "to_frame()" in text, f"{doc_name} missing to_frame()"
            assert "cf_res.shocks" in text, f"{doc_name} missing cf_res.shocks"
            assert "suite_report.passed" in text, f"{doc_name} missing suite_report.passed"

            # Obsolete attributes absent
            assert "empirical_moments" not in text, f"{doc_name} has obsolete empirical_moments"
            assert "to_dataframe()" not in text, f"{doc_name} has obsolete to_dataframe()"
            assert "implied_shocks" not in text, f"{doc_name} has obsolete implied_shocks"
            assert "all_passed" not in text, f"{doc_name} has obsolete all_passed"

    def test_all_code_blocks_have_requires_header(self, en_doc_text, es_doc_text):
        """All 5 code blocks in both EN and ES docs declare '# requires: standalone snippet'."""
        fence = re.compile(r"```python\n(.*?)```", re.S)
        for doc_name, text in [("EN", en_doc_text), ("ES", es_doc_text)]:
            blocks = fence.findall(text)
            assert len(blocks) == 5
            for i, block in enumerate(blocks):
                first_line = block.strip().splitlines()[0] if block.strip() else ""
                assert first_line == "# requires: standalone snippet", (
                    f"{doc_name} block {i} does not start with '# requires: standalone snippet': {first_line}"
                )
