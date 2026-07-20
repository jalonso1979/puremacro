"""Coverage tests for puremacro.sigma.shapley.

Tests are organized by function, covering:
- Known-value / mathematical identities
- Shape, dtype, and conservation (efficiency) properties
- Symmetry and monotonicity properties
- Error handling (pytest.raises)
- MC vs exact agreement
- Edge cases (n=1, empty subset, deltaV=0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.sigma.shapley import (
    shapley_var,
    shapley_table,
    shapley_factors,
    _build_sigma,
    _v_subset,
    _shapley_exact,
    _shapley_mc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_corr(n: int, rho: float) -> np.ndarray:
    """Equicorrelation matrix: diag=1, off-diag=rho."""
    R = np.full((n, n), rho)
    np.fill_diagonal(R, 1.0)
    return R


# ===========================================================================
# _build_sigma: internal helper
# ===========================================================================

class TestBuildSigma:
    def test_diagonal_only_with_identity(self):
        sigma = np.array([2.0, 3.0])
        R = np.eye(2)
        Sigma = _build_sigma(sigma, R)
        expected = np.diag([4.0, 9.0])
        np.testing.assert_allclose(Sigma, expected, rtol=1e-12)

    def test_off_diagonal_from_correlation(self):
        sigma = np.array([2.0, 1.0])
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        Sigma = _build_sigma(sigma, R)
        # Sigma = diag(2,1) @ [[1,.5],[.5,1]] @ diag(2,1)
        # = [[4, 1], [1, 1]]
        expected = np.array([[4.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(Sigma, expected, rtol=1e-12)

    def test_output_is_symmetric(self):
        sigma = np.array([1.0, 2.0, 0.5])
        R = np.array([[1.0, 0.3, 0.1],
                      [0.3, 1.0, 0.4],
                      [0.1, 0.4, 1.0]])
        Sigma = _build_sigma(sigma, R)
        np.testing.assert_allclose(Sigma, Sigma.T, atol=1e-12)


# ===========================================================================
# _v_subset: characteristic function
# ===========================================================================

class TestVSubset:
    def test_empty_mask_returns_zero(self):
        Sigma = np.eye(2)
        w = np.array([0.5, 0.5])
        mask = np.zeros(2, dtype=bool)
        assert _v_subset(Sigma, w, mask) == 0.0

    def test_singleton_mask_equals_w2_sigma2(self):
        Sigma = np.diag([4.0, 9.0])
        w = np.array([0.5, 0.3])
        mask = np.array([True, False])
        # v({0}) = w[0]^2 * Sigma[0,0] = 0.25 * 4 = 1.0
        assert abs(_v_subset(Sigma, w, mask) - 1.0) < 1e-12

    def test_full_mask_equals_quadratic_form(self):
        sigma = np.array([1.0, 2.0])
        R = np.array([[1.0, 0.3], [0.3, 1.0]])
        Sigma = _build_sigma(sigma, R)
        w = np.array([0.4, 0.6])
        mask = np.ones(2, dtype=bool)
        expected = float(w @ Sigma @ w)
        assert abs(_v_subset(Sigma, w, mask) - expected) < 1e-12

    def test_returns_float(self):
        Sigma = np.eye(3)
        w = np.array([0.4, 0.3, 0.3])
        mask = np.array([True, True, False])
        result = _v_subset(Sigma, w, mask)
        assert isinstance(result, float)


# ===========================================================================
# shapley_var: error handling
# ===========================================================================

class TestShapleyVarErrors:
    def test_R_shape_mismatch_raises(self):
        sigma = np.array([1.0, 2.0])
        w = np.array([0.5, 0.5])
        R_bad = np.eye(3)  # wrong shape
        with pytest.raises(ValueError, match="R must be"):
            shapley_var(sigma, R_bad, w)

    def test_sigma_length_mismatch_raises(self):
        sigma = np.array([1.0])  # length 1 but w has 2
        w = np.array([0.5, 0.5])
        R = np.eye(2)
        with pytest.raises(ValueError, match="sigma must have length"):
            shapley_var(sigma, R, w)

    def test_error_message_includes_actual_shape(self):
        sigma = np.array([1.0, 2.0])
        w = np.array([0.5, 0.5])
        R = np.eye(4)
        with pytest.raises(ValueError, match=r"\(4, 4\)"):
            shapley_var(sigma, R, w)


# ===========================================================================
# shapley_var: known-value tests (exact path, n <= 14)
# ===========================================================================

class TestShapleyVarKnownValues:
    def test_scalar_component_equals_variance(self):
        """n=1: phi[0] = sigma^2 * w^2 = V."""
        sigma = np.array([3.0])
        R = np.array([[1.0]])
        w = np.array([0.5])
        phi = shapley_var(sigma, R, w)
        expected = (3.0 * 0.5) ** 2  # 2.25
        np.testing.assert_allclose(phi, [expected], rtol=1e-12)

    def test_two_uncorrelated_exact_values(self):
        """Identity R: phi_i = w_i^2 * sigma_i^2 (no cross term)."""
        sigma = np.array([1.0, 2.0])
        R = np.eye(2)
        w = np.array([0.3, 0.7])
        phi = shapley_var(sigma, R, w)
        expected = np.array([0.09, 1.96])
        np.testing.assert_allclose(phi, expected, rtol=1e-12)

    def test_efficiency_two_components_correlated(self):
        """phi sums to V = w' Sigma w."""
        sigma = np.array([1.0, 2.0])
        R = np.array([[1.0, 0.4], [0.4, 1.0]])
        w = np.array([0.6, 0.4])
        Sigma = np.diag(sigma) @ R @ np.diag(sigma)
        V = float(w @ Sigma @ w)
        phi = shapley_var(sigma, R, w)
        np.testing.assert_allclose(phi.sum(), V, rtol=1e-10)

    def test_symmetry_equal_components(self):
        """Symmetric w, sigma: each component gets same phi."""
        n = 3
        sigma = np.ones(n)
        R = _make_corr(n, 0.5)
        w = np.ones(n) / n
        phi = shapley_var(sigma, R, w)
        # By symmetry all phi_i must be equal (V/n)
        np.testing.assert_allclose(phi, phi[0] * np.ones(n), rtol=1e-10)

    def test_efficiency_conservation_3_components(self):
        """phi.sum() == V for 3-component correlated case."""
        sigma = np.array([0.5, 1.0, 1.5])
        R = np.array([[1.0, 0.4, 0.2],
                      [0.4, 1.0, 0.5],
                      [0.2, 0.5, 1.0]])
        w = np.array([0.4, 0.3, 0.3])
        Sigma = np.diag(sigma) @ R @ np.diag(sigma)
        V = float(w @ Sigma @ w)
        phi = shapley_var(sigma, R, w)
        np.testing.assert_allclose(phi.sum(), V, rtol=1e-10)

    def test_return_se_exact_path_zeros(self):
        """Exact path returns se=0 when return_se=True."""
        sigma = np.array([1.0, 1.0])
        R = np.eye(2)
        w = np.array([0.5, 0.5])
        phi, se = shapley_var(sigma, R, w, return_se=True)
        np.testing.assert_allclose(se, np.zeros(2), atol=1e-15)

    def test_output_shape_matches_n(self):
        for n in [1, 2, 5, 10]:
            sigma = np.ones(n)
            R = np.eye(n)
            w = np.ones(n) / n
            phi = shapley_var(sigma, R, w)
            assert phi.shape == (n,)

    def test_output_dtype_float(self):
        sigma = np.array([1.0, 2.0])
        R = np.eye(2)
        w = np.array([0.5, 0.5])
        phi = shapley_var(sigma, R, w)
        assert phi.dtype.kind == "f"

    def test_nonnegative_for_psd_correlation(self):
        """For PSD Sigma and non-negative w, phi_i >= 0."""
        sigma = np.array([1.0, 2.0, 0.5])
        R = np.array([[1.0, 0.3, 0.2],
                      [0.3, 1.0, 0.1],
                      [0.2, 0.1, 1.0]])
        w = np.array([0.4, 0.4, 0.2])
        phi = shapley_var(sigma, R, w)
        assert phi.min() >= -1e-12

    def test_larger_sigma_gives_larger_phi(self):
        """Doubling one sigma (holding others fixed) must increase that component's phi."""
        sigma_low = np.array([1.0, 1.0])
        sigma_high = np.array([2.0, 1.0])
        R = np.array([[1.0, 0.4], [0.4, 1.0]])
        w = np.array([0.5, 0.5])
        phi_low = shapley_var(sigma_low, R, w)
        phi_high = shapley_var(sigma_high, R, w)
        assert phi_high[0] > phi_low[0]

    def test_perfect_correlation_efficiency(self):
        """V = (sum_i w_i * sigma_i)^2 when all corr=1."""
        sigma = np.array([1.0, 2.0])
        R = np.ones((2, 2))  # all ones -> rank-1
        w = np.array([0.3, 0.7])
        expected_V = (0.3 * 1.0 + 0.7 * 2.0) ** 2
        phi = shapley_var(sigma, R, w)
        np.testing.assert_allclose(phi.sum(), expected_V, rtol=1e-10)

    def test_accepts_list_inputs(self):
        """shapley_var should accept Python lists (converts to array)."""
        phi = shapley_var([1.0, 1.0], [[1.0, 0.0], [0.0, 1.0]], [0.5, 0.5])
        assert phi.shape == (2,)


# ===========================================================================
# shapley_var: Monte-Carlo path (explicit n_perm)
# ===========================================================================

class TestShapleyVarMC:
    def test_mc_efficiency_conservation(self):
        """MC phi sums approximately to V."""
        sigma = np.array([1.0, 2.0])
        R = np.array([[1.0, 0.4], [0.4, 1.0]])
        w = np.array([0.6, 0.4])
        Sigma = np.diag(sigma) @ R @ np.diag(sigma)
        V = float(w @ Sigma @ w)
        phi = shapley_var(sigma, R, w, n_perm=2000, seed=0)
        np.testing.assert_allclose(phi.sum(), V, rtol=1e-3)

    def test_mc_agrees_with_exact_for_small_n(self):
        """With enough permutations, MC ≈ exact for n=4."""
        sigma = np.array([1.0, 2.0, 0.5, 1.5])
        R = np.array([[1.0, 0.3, 0.1, 0.2],
                      [0.3, 1.0, 0.4, 0.1],
                      [0.1, 0.4, 1.0, 0.5],
                      [0.2, 0.1, 0.5, 1.0]])
        w = np.array([0.4, 0.3, 0.2, 0.1])
        phi_exact = shapley_var(sigma, R, w)
        phi_mc, se = shapley_var(sigma, R, w, n_perm=5000, seed=42, return_se=True)
        # Within 4 standard errors for each component
        np.testing.assert_allclose(phi_mc, phi_exact, atol=4 * se.max() + 1e-8)

    def test_mc_return_se_shapes(self):
        """return_se=True with explicit n_perm returns (phi, se) both shape (n,)."""
        n = 3
        sigma = np.ones(n)
        R = _make_corr(n, 0.3)
        w = np.ones(n) / n
        phi, se = shapley_var(sigma, R, w, n_perm=200, seed=7, return_se=True)
        assert phi.shape == (n,)
        assert se.shape == (n,)
        assert (se >= 0).all()

    def test_mc_se_positive_when_nondegenerate(self):
        """Standard errors > 0 for correlated components (genuine MC variance)."""
        sigma = np.array([1.0, 2.0, 0.5])
        R = _make_corr(3, 0.5)
        w = np.array([0.4, 0.4, 0.2])
        _, se = shapley_var(sigma, R, w, n_perm=300, seed=1, return_se=True)
        # At least one component should have se > 0
        assert se.max() > 0

    def test_mc_seed_reproducibility(self):
        """Same seed produces identical result."""
        sigma = np.array([1.0, 1.5])
        R = np.array([[1.0, 0.3], [0.3, 1.0]])
        w = np.array([0.5, 0.5])
        phi1 = shapley_var(sigma, R, w, n_perm=100, seed=99)
        phi2 = shapley_var(sigma, R, w, n_perm=100, seed=99)
        np.testing.assert_array_equal(phi1, phi2)

    def test_mc_different_seeds_differ(self):
        """Different seeds produce different (but close) results."""
        sigma = np.array([1.0, 2.0])
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        w = np.array([0.5, 0.5])
        phi1 = shapley_var(sigma, R, w, n_perm=50, seed=0)
        phi2 = shapley_var(sigma, R, w, n_perm=50, seed=1234)
        # Should not be identical
        assert not np.allclose(phi1, phi2)

    def test_auto_mc_for_large_n(self):
        """n > 14 triggers automatic MC (n_perm=2000)."""
        n = 15
        sigma = np.ones(n)
        R = np.eye(n)
        w = np.ones(n) / n
        phi = shapley_var(sigma, R, w)  # n_perm=None -> MC
        V_expected = 1.0 / n
        np.testing.assert_allclose(phi.sum(), V_expected, rtol=1e-3)
        assert phi.shape == (n,)

    def test_mc_without_return_se_returns_1d(self):
        """Without return_se, MC path returns only phi (no tuple)."""
        sigma = np.array([1.0, 1.0])
        R = np.eye(2)
        w = np.array([0.5, 0.5])
        result = shapley_var(sigma, R, w, n_perm=100, seed=0)
        assert isinstance(result, np.ndarray)


# ===========================================================================
# shapley_table
# ===========================================================================

class TestShapleyTable:
    def test_columns_present(self):
        sigma = np.array([1.0, 2.0])
        R = np.eye(2)
        w = np.array([0.3, 0.7])
        tbl = shapley_table(sigma, R, w)
        for col in ("component", "w", "sigma", "phi", "share", "se"):
            assert col in tbl.columns

    def test_auto_labels(self):
        sigma = np.array([1.0, 2.0, 0.5])
        R = np.eye(3)
        w = np.ones(3) / 3
        tbl = shapley_table(sigma, R, w)
        assert list(tbl["component"]) == ["c0", "c1", "c2"]

    def test_custom_labels(self):
        sigma = np.array([1.0, 2.0])
        R = np.eye(2)
        w = np.array([0.5, 0.5])
        tbl = shapley_table(sigma, R, w, labels=["Cons", "Inv"])
        assert list(tbl["component"]) == ["Cons", "Inv"]

    def test_shares_sum_to_one(self):
        sigma = np.array([0.5, 1.0, 1.5])
        R = np.array([[1.0, 0.4, 0.2],
                      [0.4, 1.0, 0.5],
                      [0.2, 0.5, 1.0]])
        w = np.array([0.4, 0.3, 0.3])
        tbl = shapley_table(sigma, R, w)
        np.testing.assert_allclose(tbl["share"].sum(), 1.0, rtol=1e-10)

    def test_phi_matches_shapley_var(self):
        sigma = np.array([1.0, 2.0])
        R = np.array([[1.0, 0.3], [0.3, 1.0]])
        w = np.array([0.6, 0.4])
        tbl = shapley_table(sigma, R, w, seed=0)
        phi_direct = shapley_var(sigma, R, w)
        np.testing.assert_allclose(tbl["phi"].values, phi_direct, rtol=1e-10)

    def test_returns_dataframe(self):
        sigma = np.array([1.0, 1.0])
        R = np.eye(2)
        w = np.array([0.5, 0.5])
        result = shapley_table(sigma, R, w)
        assert isinstance(result, pd.DataFrame)

    def test_nrows_equals_n(self):
        n = 4
        sigma = np.ones(n)
        R = _make_corr(n, 0.2)
        w = np.ones(n) / n
        tbl = shapley_table(sigma, R, w)
        assert len(tbl) == n

    def test_w_column_matches_input(self):
        sigma = np.array([1.0, 2.0])
        R = np.eye(2)
        w = np.array([0.3, 0.7])
        tbl = shapley_table(sigma, R, w)
        np.testing.assert_allclose(tbl["w"].values, w, rtol=1e-12)

    def test_sigma_column_matches_input(self):
        sigma = np.array([1.0, 2.0])
        R = np.eye(2)
        w = np.array([0.3, 0.7])
        tbl = shapley_table(sigma, R, w)
        np.testing.assert_allclose(tbl["sigma"].values, sigma, rtol=1e-12)

    def test_table_with_mc_path(self):
        """shapley_table with explicit n_perm invokes MC and still returns valid table."""
        sigma = np.array([1.0, 1.5, 0.5])
        R = _make_corr(3, 0.3)
        w = np.ones(3) / 3
        tbl = shapley_table(sigma, R, w, n_perm=500, seed=0)
        assert (tbl["share"] >= 0).all()
        np.testing.assert_allclose(tbl["share"].sum(), 1.0, rtol=1e-3)


# ===========================================================================
# shapley_factors
# ===========================================================================

class TestShapleyFactors:
    def _pre_post_only_sigma(self):
        """Helper: only sigma changes."""
        pre = {"sigma": np.array([1.0, 1.0]),
               "R": np.eye(2),
               "w": np.array([0.5, 0.5])}
        post = {"sigma": np.array([2.0, 1.0]),
                "R": np.eye(2),
                "w": np.array([0.5, 0.5])}
        return pre, post

    def test_keys_present(self):
        pre, post = self._pre_post_only_sigma()
        result = shapley_factors(pre, post)
        expected_keys = {"V_pre", "V_post", "deltaV",
                         "phi_w", "phi_sigma", "phi_R",
                         "share_w", "share_sigma", "share_R"}
        assert expected_keys == set(result.keys())

    def test_efficiency_phi_sum_equals_deltaV(self):
        """phi_w + phi_sigma + phi_R == deltaV."""
        pre = {"sigma": np.array([1.0, 0.5]),
               "R": np.eye(2),
               "w": np.array([0.4, 0.6])}
        post = {"sigma": np.array([2.0, 1.0]),
                "R": np.array([[1.0, 0.3], [0.3, 1.0]]),
                "w": np.array([0.3, 0.7])}
        result = shapley_factors(pre, post)
        phi_sum = result["phi_w"] + result["phi_sigma"] + result["phi_R"]
        np.testing.assert_allclose(phi_sum, result["deltaV"], rtol=1e-10)

    def test_shares_sum_to_one(self):
        """share_w + share_sigma + share_R == 1 when deltaV != 0."""
        pre, post = self._pre_post_only_sigma()
        result = shapley_factors(pre, post)
        share_sum = result["share_w"] + result["share_sigma"] + result["share_R"]
        np.testing.assert_allclose(share_sum, 1.0, rtol=1e-10)

    def test_V_values_correct(self):
        """V_pre and V_post match manual calculation."""
        pre = {"sigma": np.array([1.0, 1.0]),
               "R": np.eye(2),
               "w": np.array([0.5, 0.5])}
        post = {"sigma": np.array([1.0, 1.0]),
                "R": np.eye(2),
                "w": np.array([0.5, 0.5])}
        result = shapley_factors(pre, post)
        # V_pre = V_post = 0.5^2 + 0.5^2 = 0.5
        np.testing.assert_allclose(result["V_pre"], 0.5, rtol=1e-12)
        np.testing.assert_allclose(result["V_post"], 0.5, rtol=1e-12)

    def test_only_sigma_changes_phi_w_and_phi_R_zero(self):
        """If w and R unchanged, phi_w == phi_R == 0."""
        pre, post = self._pre_post_only_sigma()
        result = shapley_factors(pre, post)
        np.testing.assert_allclose(result["phi_w"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["phi_R"], 0.0, atol=1e-10)

    def test_only_w_changes_phi_sigma_and_phi_R_zero(self):
        """If sigma and R unchanged, phi_sigma == phi_R == 0."""
        pre = {"sigma": np.array([1.0, 1.0]),
               "R": np.eye(2),
               "w": np.array([0.5, 0.5])}
        post = {"sigma": np.array([1.0, 1.0]),
                "R": np.eye(2),
                "w": np.array([0.7, 0.3])}
        result = shapley_factors(pre, post)
        np.testing.assert_allclose(result["phi_sigma"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["phi_R"], 0.0, atol=1e-10)

    def test_only_R_changes_phi_w_and_phi_sigma_zero(self):
        """If sigma and w unchanged, phi_w == phi_sigma == 0."""
        pre = {"sigma": np.array([1.0, 1.0]),
               "R": np.eye(2),
               "w": np.array([0.5, 0.5])}
        post = {"sigma": np.array([1.0, 1.0]),
                "R": np.array([[1.0, 0.8], [0.8, 1.0]]),
                "w": np.array([0.5, 0.5])}
        result = shapley_factors(pre, post)
        np.testing.assert_allclose(result["phi_w"], 0.0, atol=1e-10)
        np.testing.assert_allclose(result["phi_sigma"], 0.0, atol=1e-10)

    def test_deltaV_equals_V_post_minus_V_pre(self):
        pre = {"sigma": np.array([1.0, 0.5]),
               "R": np.array([[1.0, 0.4], [0.4, 1.0]]),
               "w": np.array([0.6, 0.4])}
        post = {"sigma": np.array([2.0, 1.0]),
                "R": np.array([[1.0, 0.2], [0.2, 1.0]]),
                "w": np.array([0.4, 0.6])}
        result = shapley_factors(pre, post)
        np.testing.assert_allclose(
            result["deltaV"],
            result["V_post"] - result["V_pre"],
            rtol=1e-12,
        )

    def test_zero_deltaV_gives_zero_shares_but_no_error(self):
        """When V_pre == V_post, deltaV=0; shares use denom=1.0 (no division by zero)."""
        pre = {"sigma": np.array([1.0, 1.0]),
               "R": np.eye(2),
               "w": np.array([0.5, 0.5])}
        post = {"sigma": np.array([1.0, 1.0]),
                "R": np.eye(2),
                "w": np.array([0.5, 0.5])}
        result = shapley_factors(pre, post)
        assert result["deltaV"] == 0.0
        # All phi should be 0
        np.testing.assert_allclose(result["phi_w"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["phi_sigma"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["phi_R"], 0.0, atol=1e-12)

    def test_positive_deltaV_from_higher_sigma(self):
        """Increasing all sigma components increases V."""
        pre = {"sigma": np.array([1.0, 1.0]),
               "R": np.eye(2),
               "w": np.array([0.5, 0.5])}
        post = {"sigma": np.array([2.0, 2.0]),
                "R": np.eye(2),
                "w": np.array([0.5, 0.5])}
        result = shapley_factors(pre, post)
        assert result["deltaV"] > 0
        assert result["phi_sigma"] > 0

    def test_list_inputs_accepted(self):
        """shapley_factors converts list inputs to arrays without error."""
        pre = {"sigma": [1.0, 1.0], "R": [[1.0, 0.0], [0.0, 1.0]], "w": [0.5, 0.5]}
        post = {"sigma": [2.0, 1.0], "R": [[1.0, 0.0], [0.0, 1.0]], "w": [0.5, 0.5]}
        result = shapley_factors(pre, post)
        assert "deltaV" in result


# ===========================================================================
# _shapley_exact and _shapley_mc (internal): cross-check
# ===========================================================================

class TestInternalFunctions:
    def test_exact_one_component_equals_v(self):
        Sigma = np.array([[4.0]])
        w = np.array([0.5])
        phi = _shapley_exact(Sigma, w)
        assert abs(phi[0] - 1.0) < 1e-12  # (0.5)^2 * 4 = 1.0

    def test_mc_efficiency(self):
        """MC phi sums to V for a 3-component case."""
        sigma = np.array([1.0, 2.0, 0.5])
        R = _make_corr(3, 0.3)
        Sigma = np.diag(sigma) @ R @ np.diag(sigma)
        w = np.array([0.4, 0.4, 0.2])
        V = float(w @ Sigma @ w)
        rng = np.random.default_rng(0)
        phi, se = _shapley_mc(Sigma, w, n_perm=1000, rng=rng)
        np.testing.assert_allclose(phi.sum(), V, rtol=5e-3)
        assert se.shape == (3,)
        assert (se >= 0).all()

    def test_exact_equals_mc_for_n2(self):
        """For n=2, exact and MC should agree closely."""
        Sigma = np.array([[1.0, 0.5], [0.5, 4.0]])
        w = np.array([0.6, 0.4])
        phi_exact = _shapley_exact(Sigma, w)
        rng = np.random.default_rng(42)
        phi_mc, se = _shapley_mc(Sigma, w, n_perm=3000, rng=rng)
        np.testing.assert_allclose(phi_mc, phi_exact, atol=4 * se.max() + 1e-8)
