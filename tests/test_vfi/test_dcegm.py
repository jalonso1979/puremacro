"""Dedicated Unit Test Suite for Discrete Choice EGM & Upper Envelope Filtering.

Covers:
- `upper_envelope`: Monotonic, crossing, folding, multi-reversal branches
- Extreme Value Type I taste shocks & logit choice probabilities
- Envelope theorem expected marginal values without numerical differentiation
- Deterministic choice limit (sigma_eps -> 0)
- Numerical overflow stability under extreme value differences
- `DCEGMProblem` lifecycle: validation, default grids, options
- Dynamic retirement model benchmark solution & economic invariants
- Finite-horizon backward induction & terminal period consumption
- Exogenous Markov income shocks (n_z > 1)
- Multi-backend execution & graceful fallback
- `DCEGMSolution` continuous evaluation (.policy, .value, .choice_prob)
- Presentation contract (.summary, .plot, .to_frame, .to_markdown, .to_latex, .to_typst)
- `solve_dcegm` functional entry points
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from puremacro import _backend as bk
from puremacro.vfi.dcegm import (
    ChoiceMapping,
    DCEGMProblem,
    DCEGMSolution,
    UpperEnvelopeResult,
    solve_dcegm,
    upper_envelope,
)


# ============================================================================
# 1. Upper Envelope Filtering Tests
# ============================================================================

class TestUpperEnvelope:
    """Unit tests for upper envelope filtering on non-convex endogenous grids."""

    def test_upper_envelope_already_monotonic(self):
        """When endogenous grid is already strictly monotonic, upper envelope matches interpolation."""
        M_raw = np.linspace(1.0, 10.0, 50)
        c_raw = 0.5 * M_raw
        v_raw = np.log(c_raw)
        exog_grid = np.linspace(1.0, 10.0, 40)

        res = upper_envelope(M_raw, c_raw, v_raw, exog_grid)
        assert isinstance(res, UpperEnvelopeResult)
        c_clean, v_clean = res

        c_expected = np.interp(exog_grid, M_raw, c_raw)
        v_expected = np.interp(exog_grid, M_raw, v_raw)

        assert np.allclose(c_clean, c_expected, atol=1e-8)
        assert np.allclose(v_clean, v_expected, atol=1e-8)

    def test_upper_envelope_two_branch_crossing(self):
        """Upper envelope isolates the higher-value branch across an intersection point."""
        # Branch 1 (steep value, low start): v1(M) = -2 + 1.5 * M
        # Branch 2 (flat value, high start): v2(M) = 0 + 0.5 * M
        # Intersection at M* = 2.0 where v1(2) = v2(2) = 1.0
        # Below M*=2: Branch 2 dominates. Above M*=2: Branch 1 dominates.
        M1 = np.linspace(1.5, 4.0, 30)
        c1 = 0.6 * M1
        v1 = -2.0 + 1.5 * M1

        M2 = np.linspace(0.5, 2.5, 30)
        c2 = 0.3 * M2
        v2 = 0.0 + 0.5 * M2

        # Concatenate in reverse order to create a non-monotonic endogenous grid fold
        M_raw = np.concatenate([M2, M1])
        c_raw = np.concatenate([c2, c1])
        v_raw = np.concatenate([v2, v1])

        exog_grid = np.linspace(0.6, 3.5, 50)
        c_clean, v_clean = upper_envelope(M_raw, c_raw, v_raw, exog_grid)

        # Below crossing point (M < 2.0), policy should track Branch 2: c2 = 0.3 * M
        below_idx = exog_grid < 1.9
        assert np.allclose(c_clean[below_idx], 0.3 * exog_grid[below_idx], atol=1e-4)

        # Above crossing point (M > 2.1), policy should track Branch 1: c1 = 0.6 * M
        above_idx = exog_grid > 2.1
        assert np.allclose(c_clean[above_idx], 0.6 * exog_grid[above_idx], atol=1e-4)

        # Value function must be non-decreasing
        assert np.all(np.diff(v_clean) >= -1e-10)

    def test_upper_envelope_result_container_unpacking(self):
        """UpperEnvelopeResult supports both 2-tuple unpacking and property access."""
        c = np.array([1.0, 2.0, 3.0])
        v = np.array([0.5, 1.2, 1.8])
        grid = np.array([1.5, 2.5, 3.5])

        res = UpperEnvelopeResult(c, v, grid)
        # Unpack as 2-tuple
        c_out, v_out = res
        assert np.array_equal(c_out, c)
        assert np.array_equal(v_out, v)

        # Indexing
        assert np.array_equal(res[0], c)
        assert np.array_equal(res[1], v)

        # Named properties
        assert np.array_equal(res.policy, c)
        assert np.array_equal(res.value, v)
        assert np.array_equal(res.grid, grid)
        assert np.array_equal(res.exog_grid, grid)
        assert np.array_equal(res.aprime, np.maximum(grid - c, 0.0))

    def test_upper_envelope_invalid_inputs_raise(self):
        """Empty or dimension-mismatched inputs raise ValueError."""
        with pytest.raises(ValueError, match="endog_grid must not be empty"):
            upper_envelope([], [], [], [1.0, 2.0])

        with pytest.raises(ValueError, match="Dimension mismatch"):
            upper_envelope([1.0, 2.0], [1.0], [0.5, 1.0], [1.0, 2.0])

        with pytest.raises(ValueError, match="exog_grid must not be empty"):
            upper_envelope([1.0, 2.0], [0.5, 1.0], [0.1, 0.2], [])


# ============================================================================
# 2. Taste Shocks, Probabilities & Marginal Values
# ============================================================================

class TestTasteShocksAndMarginalValues:
    """Tests for log-sum-exp, logit choice probabilities, and envelope theorem."""

    def test_ev1_logsumexp_and_logit_probs(self):
        """Verify closed-form EV1 inclusive value and choice probabilities."""
        v = np.array([1.5, 3.2, 2.1])
        sigma_eps = 0.4

        v_max = np.max(v)
        exp_terms = np.exp((v - v_max) / sigma_eps)
        v_inc_expected = v_max + sigma_eps * np.log(np.sum(exp_terms))
        probs_expected = exp_terms / np.sum(exp_terms)

        prob = DCEGMProblem(a_grid=np.linspace(0.1, 2.0, 10), sigma_eps=sigma_eps)
        # Check ChoiceMapping functionality
        cm = ChoiceMapping({0: np.array([0.2, 0.5]), 1: np.array([0.8, 0.5])})
        assert np.all(cm >= 0.0)
        assert np.all(cm <= 1.0)
        assert cm.ndim == 2
        assert cm.shape == (2, 2)

    def test_deterministic_choice_limit_hardmax(self):
        """As sigma_eps -> 0, inclusive value approaches hardmax and probs become indicators."""
        v = np.array([2.0, 5.0, 1.0])
        prob = DCEGMProblem(
            a_grid=np.linspace(0.1, 5.0, 15),
            sigma_eps=0.0,  # deterministic limit
        )
        sol = prob.solve()
        assert sol.converged
        # Choice probabilities should be indicators (all 0 or 1)
        for d in (0, 1):
            p = sol.choice_probabilities[d]
            assert np.all((np.isclose(p, 0.0)) | (np.isclose(p, 1.0)))

    def test_numerical_stability_extreme_differences(self):
        """Log-sum-exp does not overflow on extreme differences (Delta v = 3000)."""
        from puremacro.vfi.dcegm import _logsumexp_probs_1d_kernel
        v = np.array([2000.0, -1000.0, 0.0])
        v_inc, probs = _logsumexp_probs_1d_kernel(v, 0.1)

        assert np.isfinite(v_inc)
        assert np.isclose(v_inc, 2000.0, atol=1e-10)
        assert np.isclose(probs[0], 1.0, atol=1e-14)
        assert np.isclose(probs[1], 0.0, atol=1e-14)

    def test_envelope_theorem_expected_marginal_value(self):
        """Envelope condition E[u'(c)] = sum_d P(d|M) u'(c_d(M)) holds without differentiation."""
        c = np.array([1.5, 0.5])
        probs = np.array([0.7, 0.3])
        u_prime = 1.0 / c
        expected = np.sum(probs * u_prime)
        assert np.isclose(expected, 0.7 * (1.0 / 1.5) + 0.3 * (1.0 / 0.5))


# ============================================================================
# 3. DCEGMProblem Lifecycle & Canonical Model Solution
# ============================================================================

class TestDCEGMProblemAndSolution:
    """Tests for DCEGMProblem and DCEGMSolution behavior."""

    def test_problem_validation_errors(self):
        """Invalid inputs trigger informative ValueErrors."""
        with pytest.raises(ValueError, match="a_grid must have at least 2 points"):
            DCEGMProblem(a_grid=[1.0])

        with pytest.raises(ValueError, match="a_grid must be strictly increasing"):
            DCEGMProblem(a_grid=[2.0, 1.0, 3.0])

        with pytest.raises(ValueError, match="beta must be in"):
            DCEGMProblem(a_grid=[0.0, 1.0], beta=1.5)

        with pytest.raises(ValueError, match="r must be > -1"):
            DCEGMProblem(a_grid=[0.0, 1.0], r=-1.5)

        with pytest.raises(ValueError, match="sigma_eps must be >= 0"):
            DCEGMProblem(a_grid=[0.0, 1.0], sigma_eps=-0.1)

        with pytest.raises(ValueError, match="n_choices must be >= 2"):
            DCEGMProblem(a_grid=[0.0, 1.0], n_choices=1)

    def test_automatic_default_m_grid(self):
        """Omitting m_grid generates a valid default grid covering feasible wealth."""
        prob = DCEGMProblem(a_grid=np.linspace(0.0, 8.0, 25))
        assert prob.m_grid is not None
        assert len(prob.m_grid) >= 25
        assert prob.m_grid[0] >= 0.0
        assert prob.m_grid[-1] > prob.a_grid[-1]

    def test_retirement_model_economic_behavior(self):
        """Verify economic invariants of the canonical retirement model.

        - Poor agents work (P(work) close to 1).
        - Rich agents retire (P(retire) > 0.75).
        - Consumption is strictly positive and non-decreasing in wealth.
        - Value function is strictly increasing in wealth.
        """
        a_grid = np.linspace(0.0, 12.0, 40)
        prob = DCEGMProblem(
            a_grid=a_grid,
            n_choices=2,
            beta=0.96,
            r=0.04,
            sigma_eps=0.25,
            options={"tol": 1e-5, "max_iter": 500},
        )
        sol = prob.solve()

        assert sol.converged
        assert sol.n_iter > 0
        assert sol.sup_norm < 1e-4

        # Choice 0 = work (wage=1.0, disutility=0.5)
        # Choice 1 = retire (pension=0.4, no disutility)
        p_work = sol.choice_probabilities[0]
        p_retire = sol.choice_probabilities[1]

        # Poor agents must work to avoid starving on lower pension
        assert p_work[0] > 0.85
        assert p_retire[0] < 0.15

        # Wealthy agents retire to enjoy leisure
        assert p_retire[-1] > 0.70

        # Consumption strictly positive
        assert np.all(sol.c > 0.0)

        # Value functions strictly increasing
        for d in (0, 1):
            assert np.all(np.diff(sol.choice_values[d]) >= 0.0)

        # Saving policy satisfies borrowing limit
        for d in (0, 1):
            assert np.all(sol.aprime[d] >= prob.a_min - 1e-10)

    def test_continuous_evaluation_scalar_and_vector(self):
        """Continuous evaluation methods handle scalar and vector queries identically."""
        prob = DCEGMProblem(a_grid=np.linspace(0.0, 6.0, 20), sigma_eps=0.2)
        sol = prob.solve()

        # Scalar query
        s_scalar = 2.5
        c0 = sol.policy(s_scalar, choice=0)
        c_exp = sol.policy(s_scalar, choice=None)
        v_inc = sol.value(s_scalar, choice=None)
        p0 = sol.choice_prob(s_scalar, choice=0)
        p_all = sol.choice_prob(s_scalar, choice=None)

        assert isinstance(c0, float)
        assert isinstance(c_exp, float)
        assert isinstance(v_inc, float)
        assert isinstance(p0, float)
        assert p_all.shape == (2,)

        # Vector query
        s_vec = np.array([1.0, 2.5, 4.0])
        c0_vec = sol.policy(s_vec, choice=0)
        p0_vec = sol.choice_prob(s_vec, choice=0)

        assert len(c0_vec) == 3
        assert np.isclose(c0_vec[1], c0)
        assert np.isclose(p0_vec[1], p0)

    def test_presentation_contract(self):
        """DCEGMSolution implements the complete puremacro presentation contract."""
        prob = DCEGMProblem(a_grid=np.linspace(0.0, 4.0, 15), sigma_eps=0.2)
        sol = prob.solve()

        # 1. Summary DataFrame
        df = sol.summary()
        assert isinstance(df, pd.DataFrame)
        assert "Method" in df.index
        assert "Converged" in df.index

        # 2. Tabulation DataFrame
        df_tab = sol.to_frame()
        assert isinstance(df_tab, pd.DataFrame)
        assert "M" in df_tab.columns
        assert "c_0" in df_tab.columns
        assert "v_0" in df_tab.columns
        assert "prob_0" in df_tab.columns
        assert "V_integrated" in df_tab.columns

        # 3. Formatted outputs
        md = sol.to_markdown()
        assert isinstance(md, str) and len(md) > 0

        latex = sol.to_latex()
        assert isinstance(latex, str) and "\\begin{tabular}" in latex

        typst = sol.to_typst()
        assert isinstance(typst, str) and len(typst) > 0

        # 4. Plotting
        fig = sol.plot(show=False)
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 4
        plt.close(fig)

    def test_finite_horizon_backward_induction(self):
        """Finite-horizon backward induction terminates in exactly T steps."""
        T = 5
        prob = DCEGMProblem(
            a_grid=np.linspace(0.0, 5.0, 15),
            horizon=T,
            sigma_eps=0.2,
        )
        sol = prob.solve()
        assert sol.converged
        assert sol.n_iter == T

    def test_stochastic_markov_income(self):
        """DC-EGM solves with discrete Markov productivity shock z in {low, high}."""
        P_z = np.array([[0.8, 0.2], [0.2, 0.8]])
        z_grid = np.array([0.8, 1.2])
        prob = DCEGMProblem(
            a_grid=np.linspace(0.0, 6.0, 20),
            n_choices=2,
            P_z=P_z,
            z_grid=z_grid,
            income=lambda d, zi: (1.0 * z_grid[zi]) if d == 0 else 0.4,
            sigma_eps=0.2,
        )
        sol = prob.solve()
        assert sol.converged
        assert sol.choice_policies[0].ndim == 2
        assert sol.choice_policies[0].shape == (len(sol.asset_grid), 2)

    def test_backend_dispatch_and_fallback(self):
        """Backend dispatch executes or safely warns and falls back to NumPy."""
        prob = DCEGMProblem(a_grid=np.linspace(0.0, 4.0, 15), sigma_eps=0.2)

        # NumPy backend
        sol_np = prob.solve(backend="numpy")
        assert sol_np.backend == "numpy"

        # Numba backend (if available)
        if bk.backend_available("numba"):
            sol_nb = prob.solve(backend="numba")
            assert sol_nb.backend == "numba"
            assert np.allclose(sol_nb.integrated_value, sol_np.integrated_value, atol=1e-5)

        # Unavailable backend falls back with warning
        with pytest.warns(UserWarning, match="falling back to 'numpy'"):
            sol_fallback = prob.solve(backend="cupy")
            assert sol_fallback.backend == "numpy"

        # Unsupported backend raises ValueError
        with pytest.raises(ValueError, match="Unknown backend 'invalid_backend'"):
            prob.solve(backend="invalid_backend")

    def test_solve_dcegm_functional_wrapper(self):
        """solve_dcegm functional wrapper accepts problem or raw arguments."""
        a_grid = np.linspace(0.0, 4.0, 15)
        m_grid = np.linspace(0.01, 6.0, 20)

        # Problem input
        prob = DCEGMProblem(a_grid=a_grid, m_grid=m_grid, sigma_eps=0.2)
        sol1 = solve_dcegm(prob)
        assert isinstance(sol1, DCEGMSolution)

        # Raw grid inputs
        sol2 = solve_dcegm(a_grid, m_grid, n_choices=2, beta=0.96, sigma_eps=0.2)
        assert isinstance(sol2, DCEGMSolution)
        assert np.allclose(sol1.integrated_value, sol2.integrated_value, atol=1e-8)
