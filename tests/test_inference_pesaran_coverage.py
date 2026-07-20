"""Coverage tests for puremacro.inference.pesaran.

Public objects tested:
  - cd_test    -- Pesaran (2021) CD statistic and p-value
  - pesaran_cd -- public alias for cd_test

Approach: known-value tests from the exact formula; property tests (shape,
dtype, bounds, sign, symmetry, permutation-invariance, monotonicity in T);
edge cases; error-handling via pytest.raises.  No network I/O; fully
self-contained with synthetic data.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from puremacro.inference.pesaran import cd_test, pesaran_cd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_panel(T: int, N: int, *, rng: np.random.Generator) -> np.ndarray:
    """Independent standard-normal panel (T x N)."""
    return rng.standard_normal((T, N))


def _correlated_panel(T: int, N: int, rho: float, *, seed: int = 42) -> np.ndarray:
    """Factor-model panel: u_it = rho*f_t + sqrt(1-rho^2)*eps_it."""
    rng = np.random.default_rng(seed)
    common = rng.standard_normal(T)
    specific = rng.standard_normal((T, N))
    return rho * common[:, None] + np.sqrt(1 - rho ** 2) * specific


# ---------------------------------------------------------------------------
# Return types and basic contract
# ---------------------------------------------------------------------------

class TestReturnTypes:
    def test_returns_tuple_of_two(self):
        panel = _make_panel(20, 4, rng=np.random.default_rng(0))
        result = cd_test(panel)
        assert len(result) == 2

    def test_cd_stat_is_float(self):
        panel = _make_panel(20, 4, rng=np.random.default_rng(1))
        stat, _ = cd_test(panel)
        assert isinstance(stat, float)

    def test_p_value_is_float(self):
        panel = _make_panel(20, 4, rng=np.random.default_rng(2))
        _, pval = cd_test(panel)
        assert isinstance(pval, float)

    def test_p_value_in_unit_interval(self):
        panel = _make_panel(50, 5, rng=np.random.default_rng(3))
        _, pval = cd_test(panel)
        assert 0.0 <= pval <= 1.0

    def test_p_value_in_unit_interval_positive_dependence(self):
        """p-value stays in [0, 1] even when stat is very large."""
        rng = np.random.default_rng(10)
        common = rng.standard_normal(100)
        panel = np.column_stack([common + 0.01 * rng.standard_normal(100)
                                 for _ in range(6)])
        _, pval = cd_test(panel)
        assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# Known-value tests — exact formula verification
# ---------------------------------------------------------------------------

class TestKnownValue:
    def test_perfect_positive_correlation_exact(self):
        """All columns identical => all pairwise rho = 1.
        CD = sqrt(2T / (N*(N-1))) * N*(N-1)/2 = sqrt(T*N*(N-1)/2).
        """
        T, N = 100, 5
        rng = np.random.default_rng(42)
        base = rng.standard_normal(T)
        panel = np.column_stack([base] * N)
        stat, _ = cd_test(panel)
        theoretical = np.sqrt(2 * T / (N * (N - 1))) * (N * (N - 1) / 2)
        assert np.isclose(stat, theoretical, atol=1e-10)

    def test_perfect_positive_correlation_p_value_zero(self):
        """Under perfect correlation stat is huge; p-value should be 0."""
        T, N = 100, 5
        rng = np.random.default_rng(43)
        base = rng.standard_normal(T)
        panel = np.column_stack([base] * N)
        _, pval = cd_test(panel)
        assert pval == pytest.approx(0.0, abs=1e-10)

    def test_perfect_negative_correlation_two_units(self):
        """N=2, panel = [x, -x]: rho = -1.
        CD = sqrt(2T / (2*1)) * (-1) = -sqrt(T).
        """
        T = 100
        rng = np.random.default_rng(44)
        x = rng.standard_normal(T)
        panel = np.column_stack([x, -x])
        stat, _ = cd_test(panel)
        theoretical = -np.sqrt(float(T))
        assert np.isclose(stat, theoretical, atol=1e-10)

    def test_two_unit_formula_cd_equals_sqrt_T_times_rho(self):
        """N=2: CD = sqrt(2T / (N*(N-1))) * rho[0,1] = sqrt(T) * rho."""
        T = 100
        rng = np.random.default_rng(55)
        x = rng.standard_normal(T)
        y = 0.7 * x + np.sqrt(1 - 0.49) * rng.standard_normal(T)
        panel = np.column_stack([x, y])
        stat, _ = cd_test(panel)
        rho = np.corrcoef(panel.T)[0, 1]
        expected = np.sqrt(float(T)) * rho
        assert np.isclose(stat, expected, atol=1e-10)

    def test_formula_three_units_exact(self):
        """With N=3 units and known pairwise correlations from corrcoef:
        CD = sqrt(2T/6) * (rho12 + rho13 + rho23).
        We read the correlation matrix ourselves and verify the formula.
        """
        T, N = 80, 3
        rng = np.random.default_rng(77)
        panel = rng.standard_normal((T, N))
        corr = np.corrcoef(panel.T)
        rho_sum = corr[0, 1] + corr[0, 2] + corr[1, 2]
        expected = np.sqrt(2 * T / (N * (N - 1))) * rho_sum
        stat, _ = cd_test(panel)
        assert np.isclose(stat, expected, atol=1e-10)

    def test_p_value_matches_two_tailed_normal(self):
        """p-value = 2*(1 - Phi(|CD|)) for any valid panel."""
        rng = np.random.default_rng(88)
        panel = rng.standard_normal((60, 4))
        stat, pval = cd_test(panel)
        expected_pval = 2.0 * (1.0 - norm.cdf(abs(stat)))
        assert np.isclose(pval, expected_pval, atol=1e-12)

    def test_zero_correlation_panel_small_stat(self):
        """With T large and truly independent columns the statistic should
        be approximately standard normal.  |stat| < 4 with overwhelming
        probability for T=300, N=8."""
        rng = np.random.default_rng(123)
        panel = rng.standard_normal((300, 8))
        stat, _ = cd_test(panel)
        assert abs(stat) < 4.0


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------

class TestProperties:
    def test_permutation_invariance(self):
        """Reordering columns does not change the CD statistic."""
        rng = np.random.default_rng(200)
        panel = rng.standard_normal((40, 5))
        stat_orig, _ = cd_test(panel)
        perm = [3, 0, 4, 1, 2]
        stat_perm, _ = cd_test(panel[:, perm])
        assert np.isclose(stat_orig, stat_perm, atol=1e-10)

    def test_positive_dependence_yields_positive_stat(self):
        """Strongly positively correlated panel => CD > 0."""
        panel = _correlated_panel(200, 6, rho=0.8, seed=300)
        stat, _ = cd_test(panel)
        assert stat > 0.0

    def test_positive_dependence_rejects_h0(self):
        """Strong dependence (rho=0.8) should reject H0 at 1% with T=200."""
        panel = _correlated_panel(200, 6, rho=0.8, seed=301)
        _, pval = cd_test(panel)
        assert pval < 0.01

    def test_larger_T_raises_abs_stat_under_alternative(self):
        """Under a fixed correlated DGP, more time periods -> larger |CD|."""
        stat_small, _ = cd_test(_correlated_panel(50, 5, rho=0.5, seed=400))
        stat_large, _ = cd_test(_correlated_panel(500, 5, rho=0.5, seed=401))
        assert abs(stat_large) > abs(stat_small)

    def test_stat_and_pval_are_finite(self):
        """No NaN or inf should appear in normal usage."""
        rng = np.random.default_rng(500)
        panel = rng.standard_normal((50, 6))
        stat, pval = cd_test(panel)
        assert np.isfinite(stat)
        assert np.isfinite(pval)

    def test_sign_of_stat_tracks_sign_of_average_correlation(self):
        """If the average pairwise correlation is positive, CD > 0; if negative, CD < 0."""
        rng = np.random.default_rng(600)
        T, N = 200, 4
        common = rng.standard_normal(T)
        # positive dependence
        pos_panel = np.column_stack([common + 0.05 * rng.standard_normal(T)
                                     for _ in range(N)])
        stat_pos, _ = cd_test(pos_panel)
        assert stat_pos > 0.0

        # negative / mixed dependence: alternate signs
        rng2 = np.random.default_rng(601)
        common2 = rng2.standard_normal(T)
        neg_panel = np.column_stack([(-1) ** i * common2 + 0.05 * rng2.standard_normal(T)
                                     for i in range(N)])
        stat_neg, _ = cd_test(neg_panel)
        # For N=4, alternating signs: (12): -1, (13): +1, (14): -1, (23): -1, (24): +1, (34): -1
        # net rho_sum < 0 => stat < 0
        assert stat_neg < 0.0


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------

class TestInputHandling:
    def test_list_of_lists_accepted(self):
        """Python list should be coerced via np.asarray."""
        lst = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        stat, pval = cd_test(lst)
        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_list_matches_numpy_array(self):
        lst = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        stat_lst, pval_lst = cd_test(lst)
        stat_arr, pval_arr = cd_test(np.array(lst))
        assert np.isclose(stat_lst, stat_arr)
        assert np.isclose(pval_lst, pval_arr)

    def test_float32_array_accepted(self):
        rng = np.random.default_rng(700)
        panel = rng.standard_normal((30, 4)).astype(np.float32)
        stat, pval = cd_test(panel)
        assert isinstance(stat, float)
        assert isinstance(pval, float)

    def test_minimum_valid_shape_t2_n2(self):
        """T=2, N=2 is the smallest valid 2D panel; must not raise."""
        panel = np.array([[1.0, 2.0], [-1.0, -2.0]])
        stat, pval = cd_test(panel)
        assert isinstance(stat, float)
        assert 0.0 <= pval <= 1.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_1d_array_raises_assertion_error(self):
        with pytest.raises(AssertionError, match="2D"):
            cd_test(np.ones(10))

    def test_3d_array_raises_assertion_error(self):
        with pytest.raises(AssertionError, match="2D"):
            cd_test(np.ones((5, 3, 2)))

    def test_scalar_raises_assertion_error(self):
        with pytest.raises((AssertionError, AttributeError)):
            cd_test(42.0)


# ---------------------------------------------------------------------------
# pesaran_cd alias
# ---------------------------------------------------------------------------

class TestAlias:
    def test_pesaran_cd_is_callable(self):
        assert callable(pesaran_cd)

    def test_pesaran_cd_matches_cd_test_output(self):
        rng = np.random.default_rng(800)
        panel = rng.standard_normal((40, 5))
        stat1, pval1 = cd_test(panel)
        stat2, pval2 = pesaran_cd(panel)
        assert stat1 == stat2
        assert pval1 == pval2

    def test_pesaran_cd_alias_is_same_object(self):
        """The alias should literally be the same function (not a wrapper)."""
        assert pesaran_cd is cd_test


# ---------------------------------------------------------------------------
# Size (false-positive rate) under the null — slow test
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestSizeUnderNull:
    def test_empirical_rejection_rate_near_5pct(self):
        """Run 1000 Monte-Carlo trials under H0 (IID N(0,1) residuals).
        The empirical rejection rate at nominal 5% should be within
        [2%, 8%] — covers minor finite-sample distortions.
        """
        T, N, n_sim = 200, 6, 1000
        rng = np.random.default_rng(9999)
        rejections = sum(
            1 for _ in range(n_sim)
            if cd_test(rng.standard_normal((T, N)))[1] < 0.05
        )
        rate = rejections / n_sim
        assert 0.02 <= rate <= 0.08, (
            f"Empirical rejection rate {rate:.3f} is far from 5%; "
            "cd_test may be mis-calibrated."
        )
