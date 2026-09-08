"""Tests for puremacro.dsge.steady — robust block solver, DM diagnosis, and homotopy.

Validations:
- Validation 6: RBC and SW07 steady state matches hybr bit-identically.
- Validation 7: 50-equation recursive model where hybr from default guess fails
  and block solver succeeds.
- Validation 8: Dulmage-Mendelsohn report on an overdetermined/structurally
  singular specification accurately identifies over- and under-determined subsets
  without unhandled exceptions.
- Homotopy continuation with adaptive bisection converges along a parameter path
  where a single direct solve fails.
- Solver menu ("block", "hybr", "lm", "df-sane").
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

import puremacro.dsge as dsge
from puremacro.dsge.build import SteadyStateError, build
from puremacro.dsge.dynare import load_mod, parse_mod
from puremacro.dsge.steady import (
    StructuralSingularityError,
    _Vec,
    dulmage_mendelsohn,
    hopcroft_karp,
    numeric_incidence_matrix,
    steady,
    tarjan_scc,
)


# ===========================================================================
# Validation 6: RBC and SW07 bit-identical between block and hybr
# ===========================================================================

def growth_equations(xp, x, e, p):
    """Canonical neoclassical growth / RBC equations."""
    return [
        1.0 / x.c - p.beta * (p.alpha * xp.z * xp.k ** (p.alpha - 1.0)) / xp.c,
        x.c + xp.k - x.z * x.k ** p.alpha,
        xp.z - x.z ** p.rho * np.exp(e.eps),
    ]


def test_validation_6_rbc_steady_state_bit_identical():
    """Validation 6a: RBC steady state from block solver matches hybr bit-identically."""
    params = dict(alpha=0.33, beta=0.98, rho=0.90)
    guess = dict(c=0.5, k=0.1, z=1.0)
    variables = ["c", "k", "z"]

    ss_block, info_block = steady(
        growth_equations,
        variables,
        guess,
        params,
        shocks=["eps"],
        solve_algo="block",
    )
    ss_hybr, info_hybr = steady(
        growth_equations,
        variables,
        guess,
        params,
        shocks=["eps"],
        solve_algo="hybr",
    )

    assert info_block["converged"]
    assert info_hybr["converged"]
    assert info_block["max_residual"] < 1e-12
    assert info_hybr["max_residual"] < 1e-12

    # Must be bit-identical
    assert np.array_equal(ss_block, ss_hybr)
    assert np.max(np.abs(ss_block - ss_hybr)) == 0.0

    # Also check end-to-end via dsge.build()
    m_block = build(
        growth_equations,
        variables=variables,
        states=["k", "z"],
        shocks=["eps"],
        params=params,
        guess=guess,
        solve_algo="block",
    )
    m_hybr = build(
        growth_equations,
        variables=variables,
        states=["k", "z"],
        shocks=["eps"],
        params=params,
        guess=guess,
        solve_algo="hybr",
    )

    assert np.array_equal(m_block.steady_state.to_numpy(), m_hybr.steady_state.to_numpy())
    assert np.array_equal(m_block.solution.G, m_hybr.solution.G)
    assert np.array_equal(m_block.solution.F, m_hybr.solution.F)


def test_validation_6_sw07_steady_state_bit_identical():
    """Validation 6b: SW07 steady state from block solver matches hybr bit-identically."""
    mod_path = Path(dsge.__file__).parent / "_references" / "sw07_pfeifer.mod"
    assert mod_path.is_file(), f"SW07 reference file missing: {mod_path}"

    parsed = parse_mod(mod_path.read_text())
    guess_dict = {v: 0.0 for v in parsed["variables"]}
    if parsed["steady_state"]:
        guess_dict.update(parsed["steady_state"])

    ss_block, info_block = steady(
        parsed["equations"],
        parsed["variables"],
        guess_dict,
        parsed["params"],
        shocks=parsed["shocks"],
        solve_algo="block",
    )
    ss_hybr, info_hybr = steady(
        parsed["equations"],
        parsed["variables"],
        guess_dict,
        parsed["params"],
        shocks=parsed["shocks"],
        solve_algo="hybr",
    )

    assert info_block["converged"]
    assert info_hybr["converged"]
    assert info_block["max_residual"] < 1e-8
    assert info_hybr["max_residual"] < 1e-8

    # Must be bit-identical
    assert np.array_equal(ss_block, ss_hybr)
    assert np.max(np.abs(ss_block - ss_hybr)) == 0.0


# ===========================================================================
# Validation 7: 50-equation recursive model where hybr fails and block succeeds
# ===========================================================================

def test_validation_7_50_equation_recursive_model_block_vs_hybr():
    """Validation 7: 50-equation recursive model where hybr from default guess fails and block succeeds."""
    n_eqs = 50
    variables = [f"x{i}" for i in range(n_eqs)]

    def recursive_50_eqs(xp, x, e, p):
        # x0 is rooted at 10.0; each subsequent variable xi satisfies:
        # xi^3 - 5*xi + x_{i-1} - 2 = 0
        res = [x[0] - 10.0]
        for i in range(1, n_eqs):
            res.append(x[i] ** 3 - 5.0 * x[i] + x[i - 1] - 2.0)
        return res

    default_guess = {v: 1.0 for v in variables}

    # 1. Full-system hybr from default guess fails to converge
    with pytest.raises(SteadyStateError) as exc_info:
        steady(
            recursive_50_eqs,
            variables,
            default_guess,
            None,
            solve_algo="hybr",
        )
    assert "steady state did not converge" in str(exc_info.value)

    # 2. Block-triangular solver succeeds and decomposes into 50 singleton blocks
    ss_block, info = steady(
        recursive_50_eqs,
        variables,
        default_guess,
        None,
        solve_algo="block",
    )

    assert info["converged"]
    assert info["n_blocks"] == 50
    assert info["max_residual"] < 1e-8

    # Verify that the returned vector solves the system
    r = np.asarray(recursive_50_eqs(ss_block, ss_block, None, None), dtype=float)
    assert np.max(np.abs(r)) < 1e-8
    assert ss_block[0] == pytest.approx(10.0, abs=1e-8)


# ===========================================================================
# Validation 8: Dulmage-Mendelsohn structural singularity report
# ===========================================================================

def test_validation_8_dulmage_mendelsohn_structural_singularity():
    """Validation 8: Dulmage-Mendelsohn report accurately identifies over- and under-determined subsets."""
    # 3 equations, 3 variables (x, y, z).
    # eq_1: x - 1 = 0
    # eq_2: x + 2 = 0  (conflicts/overdetermines x with eq_1)
    # eq_3: x + y = 0
    # z is underdetermined (never appears in any equation)
    def singular_eqs(xp, x, e, p):
        return [
            x.x - 1.0,
            x.x + 2.0,
            x.x + x.y,
        ]

    variables = ["x", "y", "z"]
    guess = {"x": 1.0, "y": 1.0, "z": 1.0}
    eq_names = ["eq_over1", "eq_over2", "eq_well"]

    with pytest.raises(StructuralSingularityError) as exc_info:
        steady(
            singular_eqs,
            variables,
            guess,
            None,
            equation_names=eq_names,
            solve_algo="block",
        )

    err = exc_info.value
    assert isinstance(err, SteadyStateError)
    assert err.matching_size == 2
    assert err.n_equations == 3
    assert err.n_variables == 3

    # Dulmage-Mendelsohn diagnosis subsets
    assert set(err.overdetermined_equations) == {"eq_over1", "eq_over2"}
    assert set(err.overdetermined_variables) == {"x"}
    assert set(err.underdetermined_variables) == {"z"}
    assert len(err.underdetermined_equations) == 0

    # Error message contains informative block diagnosis
    msg = str(err)
    assert "structurally singular in steady state" in msg
    assert "Over-determined block" in msg
    assert "Under-determined block" in msg
    assert "eq_over1" in msg or "eq_over2" in msg
    assert "x" in msg
    assert "z" in msg


def test_validation_8_dulmage_mendelsohn_propagates_through_build():
    """Validation 8b: StructuralSingularityError propagates through dsge.build()."""
    def singular_model(xp, x, e, p):
        return [
            x.k - 1.0,
            x.k - 2.0,
            x.c - 0.5 * x.k,
        ]

    with pytest.raises(StructuralSingularityError) as exc_info:
        build(
            singular_model,
            variables=["k", "c", "unused_var"],
            states=["k"],
            shocks=["eps"],
            guess={"k": 1.0, "c": 1.0, "unused_var": 0.0},
            solve_algo="block",
        )

    err = exc_info.value
    assert "unused_var" in err.underdetermined_variables
    assert "k" in err.overdetermined_variables


# ===========================================================================
# Homotopy continuation with adaptive bisection
# ===========================================================================

def test_homotopy_continuation_adaptive_bisection_converges_where_direct_fails():
    """Homotopy continuation with adaptive bisection converges along parameter path where single direct solve fails."""
    # Growth model with technology parameter A shifted from 1.0 to 50.0.
    # At A=50.0, steady-state k is ~1213, c is ~399.
    # From guess c=1.0, k=3.0, direct solve at A=50.0 fails due to large step / non-convergence.
    def growth_model_A(xp, x, e, p):
        A = p.A
        alpha = p.alpha
        beta = p.beta
        delta = p.delta
        return [
            1.0 / x.c - beta * (1.0 / xp.c) * (alpha * A * xp.k ** (alpha - 1.0) + 1.0 - delta),
            x.c + xp.k - (A * x.k ** alpha + (1.0 - delta) * x.k),
        ]

    params_target = {"A": 50.0, "alpha": 0.33, "beta": 0.96, "delta": 0.1}
    guess = {"c": 1.0, "k": 3.0}

    # Direct solve fails
    with pytest.raises(SteadyStateError):
        steady(
            growth_model_A,
            ["c", "k"],
            guess,
            params_target,
            solve_algo="hybr",
        )

    # Homotopy continuation succeeds with adaptive bisection
    ss_hom, info = steady(
        growth_model_A,
        ["c", "k"],
        guess,
        params_target,
        homotopy={"A": (1.0, 50.0)},
        homotopy_steps=5,
        solve_algo="hybr",
    )

    assert info["converged"]
    assert info["homotopy_steps"] > 0
    assert info["bisections"] > 0
    assert info["max_residual"] < 1e-8

    # Verify steady-state values satisfy analytical closed-form:
    # 1/beta - 1 + delta = alpha * A * k^(alpha - 1)
    # => k_ss = ((1/beta - 1 + delta) / (alpha * A))^(1 / (alpha - 1))
    # => c_ss = A * k_ss^alpha - delta * k_ss
    alpha, beta, delta, A = 0.33, 0.96, 0.1, 50.0
    r_k = (1.0 / beta - 1.0 + delta) / (alpha * A)
    k_expected = r_k ** (1.0 / (alpha - 1.0))
    c_expected = A * k_expected ** alpha - delta * k_expected

    assert ss_hom[0] == pytest.approx(c_expected, rel=1e-6)
    assert ss_hom[1] == pytest.approx(k_expected, rel=1e-6)


def test_homotopy_continuation_stalled_raises_informative_error():
    """Homotopy continuation raises SteadyStateError when step size falls below min_step."""
    # Infeasible target parameter
    def impossible_eq(xp, x, e, p):
        return [x.x - np.sqrt(p.val)]

    # Negative val is invalid
    with pytest.raises(SteadyStateError) as exc_info:
        steady(
            impossible_eq,
            ["x"],
            {"x": 1.0},
            {"val": -10.0},
            homotopy={"val": (1.0, -10.0)},
            homotopy_steps=2,
        )

    assert "Homotopy continuation stalled" in str(exc_info.value) or "Homotopy initialization failed" in str(exc_info.value)


# ===========================================================================
# Solver menu: block, hybr, lm, df-sane
# ===========================================================================

@pytest.mark.parametrize("algo", ["block", "hybr", "lm", "df-sane"])
def test_solver_menu_all_algorithms(algo):
    """Verify all 4 solvers in the menu solve a standard nonlinear system."""
    def test_system(x, p=None):
        return [
            x[0] ** 3 - 1.0,
            x[1] ** 3 - 8.0,
            x[2] + x[1] - 3.0,
        ]

    variables = ["x0", "x1", "x2"]
    guess = [0.5, 1.5, 0.5]

    ss, info = steady(test_system, variables, guess, None, solve_algo=algo, tol=1e-6)
    assert info["converged"]
    assert info["solve_algo"] == algo
    assert info["max_residual"] < 1e-6
    assert np.allclose(ss, [1.0, 2.0, 1.0], atol=1e-5)


def test_solver_menu_invalid_algo_raises_value_error():
    """Passing an unsupported algorithm name raises ValueError."""
    with pytest.raises(ValueError, match="unknown solve_algo"):
        steady(
            lambda x: [x[0]],
            ["x"],
            [1.0],
            solve_algo="nonexistent_solver",
        )


# ===========================================================================
# Algorithmic component unit tests
# ===========================================================================

def test_hopcroft_karp_unit_cases():
    """Unit test Hopcroft-Karp on canonical graph structures."""
    # 1. Complete matching on 3x3 identity
    adj_id = [[0], [1], [2]]
    pu, pv = hopcroft_karp(3, 3, adj_id)
    assert pu == {0: 0, 1: 1, 2: 2}
    assert pv == {0: 0, 1: 1, 2: 2}

    # 2. Complete matching on complete bipartite graph
    adj_complete = [[0, 1, 2], [0, 1, 2], [0, 1, 2]]
    pu, pv = hopcroft_karp(3, 3, adj_complete)
    assert all(v is not None for v in pu.values())

    # 3. Partial matching (bottleneck)
    adj_bottle = [[0], [0], [1]]  # 3 equations, 2 variables
    pu, pv = hopcroft_karp(3, 2, adj_bottle)
    matched_u = sum(1 for v in pu.values() if v is not None)
    assert matched_u == 2


def test_dulmage_mendelsohn_unit_cases():
    """Unit test Dulmage-Mendelsohn decomposition."""
    adj = [[0], [0], [1, 2]]
    pu, pv = hopcroft_karp(3, 3, adj)
    dm = dulmage_mendelsohn(3, 3, adj, pu, pv)

    # u0 and u1 both compete for v0 -> overdetermined
    assert 0 in dm["overdetermined_variables"]
    assert (0 in dm["overdetermined_equations"] and 1 in dm["overdetermined_equations"])


def test_tarjan_scc_unit_cases():
    """Unit test Tarjan SCC on topological graphs."""
    # Simple 3-node DAG: 2 -> 1 -> 0
    # Expected topological order: [0], [1], [2]
    dag = [[], [0], [1]]
    sccs = tarjan_scc(3, dag)
    assert sccs == [[0], [1], [2]]

    # 2-node cycle with 1 isolated node: (0 <-> 1), 2
    cycle_graph = [[1], [0], []]
    sccs_cycle = tarjan_scc(3, cycle_graph)
    # The SCC containing {0, 1} and the SCC containing {2}
    scc_sets = [set(s) for s in sccs_cycle]
    assert {0, 1} in scc_sets
    assert {2} in scc_sets


def test_numeric_incidence_matrix_detects_dependencies():
    """Verify numeric incidence matrix detects genuine variable dependencies."""
    def res_fn(x):
        return np.array([
            x[0] ** 2 + 2.0 * x[1],
            x[1] - 5.0,
            x[2] ** 3,
        ])

    x0 = np.array([1.0, 2.0, 3.0])
    inc = numeric_incidence_matrix(res_fn, x0)
    assert inc.shape == (3, 3)

    # eq 0 depends on x0 and x1, not x2
    assert inc[0, 0] and inc[0, 1]
    assert not inc[0, 2]

    # eq 1 depends on x1 only
    assert inc[1, 1]
    assert not inc[1, 0] and not inc[1, 2]

    # eq 2 depends on x2 only
    assert inc[2, 2]
    assert not inc[2, 0] and not inc[2, 1]


def test_named_vector_vec_contract():
    """Verify _Vec container behaves cleanly with names, keys, slices, and errors."""
    v = _Vec(["c", "k", "z"], [1.0, 2.0, 3.0], "variable")
    assert v.c == 1.0
    assert v["k"] == 2.0
    assert v[2] == 3.0
    assert list(v) == [1.0, 2.0, 3.0]
    assert len(v) == 3

    with pytest.raises(AttributeError, match="no variable named 'nonexistent'"):
        _ = v.nonexistent

    with pytest.raises(KeyError, match="no variable named 'nonexistent'"):
        _ = v["nonexistent"]


def test_steady_input_validation():
    """Verify steady() catches bad inputs: dimension mismatch, missing guess keys."""
    def dummy_eq(x):
        return [x[0]]

    with pytest.raises(SteadyStateError, match="guess length 2 does not match"):
        steady(dummy_eq, ["x"], [1.0, 2.0])

    with pytest.raises(SteadyStateError, match="guess is missing values"):
        steady(dummy_eq, ["x", "y"], {"x": 1.0})


def test_steady_equation_names_length_mismatch_raises_value_error():
    """Verify steady() raises ValueError when len(equation_names) != len(variables)."""
    def dummy_eq(x):
        return [x[0]]

    with pytest.raises(
        ValueError,
        match="equation_names length 2 does not match number of variables 1",
    ):
        steady(dummy_eq, ["x"], [1.0], equation_names=["eq1", "eq2"])

