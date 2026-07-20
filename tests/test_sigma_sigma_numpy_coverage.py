"""Coverage tests for puremacro.sigma.sigma_numpy.SigmaObject.

This is the simpler/earlier SigmaObject (Python port of MAV/SigmaObject.m)
living in puremacro.sigma — distinct from the richer puremacro.volatility.sigma
version. Tests are organised by function/branch:

- Construction: Sigma assembly, symmetrisation of R, PSD enforcement
- _clip_psd: known eigenvalue-clipping behaviour
- var / var_no_cov: quadratic-form known values + conservation
- cov_premium_var: zero-denominator branch + positive case
- mean_corr: n=1 edge-case (off.size == 0) + general
- diag_contrib: shape, dtype, conservation
- psd_enforce=False path
- Optional attributes: country, period, labels
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.sigma.sigma_numpy import SigmaObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equicorr(n: int, rho: float) -> np.ndarray:
    """Equicorrelation matrix: diag=1, off-diag=rho."""
    R = np.full((n, n), rho)
    np.fill_diagonal(R, 1.0)
    return R


def _make_object(n: int = 2, rho: float = 0.3) -> SigmaObject:
    """Quick factory: unit sigmas, equicorrelation."""
    sigma = np.ones(n)
    R = _equicorr(n, rho)
    labels = [f"c{i}" for i in range(n)]
    return SigmaObject(sigma=sigma, R=R, labels=labels)


# ===========================================================================
# Construction and __post_init__
# ===========================================================================

class TestConstruction:
    def test_sigma_stored_as_float_array(self):
        S = SigmaObject(sigma=[1, 2], R=np.eye(2), labels=["a", "b"])
        assert S.sigma.dtype == float
        assert S.sigma.shape == (2,)

    def test_R_stored_as_float_array(self):
        S = SigmaObject(sigma=[1.0, 1.0], R=[[1, 0], [0, 1]], labels=["a", "b"])
        assert S.R.dtype == float

    def test_Sigma_shape_matches_n(self):
        for n in [1, 2, 3, 5]:
            sigma = np.ones(n)
            R = np.eye(n)
            labels = [str(i) for i in range(n)]
            S = SigmaObject(sigma=sigma, R=R, labels=labels)
            assert S.Sigma.shape == (n, n)

    def test_Sigma_is_symmetric(self):
        sigma = np.array([2.0, 1.0, 0.5])
        R = np.array([[1.0, 0.4, 0.2],
                      [0.4, 1.0, 0.3],
                      [0.2, 0.3, 1.0]])
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b", "c"])
        np.testing.assert_allclose(S.Sigma, S.Sigma.T, atol=1e-14)

    def test_asymmetric_R_gets_symmetrised(self):
        """R = [[1, 0.6],[0.4, 1]] must be symmetrised to [[1, 0.5],[0.5, 1]]."""
        R_asym = np.array([[1.0, 0.6], [0.4, 1.0]])
        S = SigmaObject(sigma=np.array([1.0, 1.0]), R=R_asym, labels=["a", "b"])
        np.testing.assert_allclose(S.R[0, 1], 0.5, atol=1e-14)
        np.testing.assert_allclose(S.R[1, 0], 0.5, atol=1e-14)

    def test_Sigma_known_value_2x2(self):
        """Sigma = diag(sigma) @ R @ diag(sigma):
        sigma=[2,1], R=[[1,.5],[.5,1]] -> [[4,1],[1,1]]."""
        sigma = np.array([2.0, 1.0])
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        expected = np.array([[4.0, 1.0], [1.0, 1.0]])
        np.testing.assert_allclose(S.Sigma, expected, rtol=1e-12)

    def test_identity_R_gives_diagonal_Sigma(self):
        sigma = np.array([3.0, 4.0])
        R = np.eye(2)
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        expected = np.diag([9.0, 16.0])
        np.testing.assert_allclose(S.Sigma, expected, rtol=1e-12)

    def test_optional_country_period_attributes(self):
        S = SigmaObject(sigma=np.array([1.0, 1.0]), R=np.eye(2),
                        labels=["a", "b"], country="USA", period="2020Q1")
        assert S.country == "USA"
        assert S.period == "2020Q1"

    def test_default_country_period_empty_string(self):
        S = _make_object(n=2)
        assert S.country == ""
        assert S.period == ""

    def test_psd_enforce_default_true(self):
        S = _make_object(n=2, rho=0.3)
        assert S.psd_enforce is True

    def test_Sigma_is_psd_after_construction(self):
        """PSD enforcement must clip any negative eigenvalues to >= tol."""
        # Indefinite R: eigenvalues -1 and 3
        R_indef = np.array([[1.0, 2.0], [2.0, 1.0]])
        S = SigmaObject(sigma=np.array([1.0, 1.0]), R=R_indef, labels=["a", "b"])
        eigs = np.linalg.eigvalsh(S.Sigma)
        assert eigs.min() >= S.psd_tol - 1e-12

    def test_psd_enforce_false_skips_clipping(self):
        """With psd_enforce=False, indefinite Sigma may have negative eigenvalues."""
        R_indef = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigs: -1 and 3
        S = SigmaObject(sigma=np.array([1.0, 1.0]), R=R_indef, labels=["a", "b"],
                        psd_enforce=False)
        eigs = np.linalg.eigvalsh(S.Sigma)
        # At least one eigenvalue should be negative (indefinite input preserved)
        assert eigs.min() < 0

    def test_labels_stored_as_sequence(self):
        S = SigmaObject(sigma=np.array([1.0, 1.0]), R=np.eye(2),
                        labels=["X", "Y"])
        assert list(S.labels) == ["X", "Y"]


# ===========================================================================
# _clip_psd
# ===========================================================================

class TestClipPSD:
    def test_clips_negative_eigenvalue_to_tol(self):
        """M = [[1,2],[2,1]] has eigenvalues 3 and -1; clipped to tol."""
        M = np.array([[1.0, 2.0], [2.0, 1.0]])
        clipped = SigmaObject._clip_psd(M, 1e-10)
        eigs = np.linalg.eigvalsh(clipped)
        assert eigs.min() >= 1e-10 - 1e-14

    def test_psd_input_unchanged_within_tolerance(self):
        """Already-PSD matrix eigenvalues should not change materially."""
        M = np.diag([4.0, 9.0])
        clipped = SigmaObject._clip_psd(M, 1e-10)
        np.testing.assert_allclose(clipped, M, rtol=1e-12)

    def test_output_is_symmetric(self):
        M = np.array([[2.0, 1.5], [1.5, 1.0]])
        clipped = SigmaObject._clip_psd(M, 1e-10)
        np.testing.assert_allclose(clipped, clipped.T, atol=1e-14)

    def test_all_zero_eigenvalues_produces_zero_matrix(self):
        """Zero matrix stays zero (all eigenvalues = 0, clipped to tol >= 0)."""
        M = np.zeros((3, 3))
        clipped = SigmaObject._clip_psd(M, 0.0)  # tol=0: clip to 0
        eigs = np.linalg.eigvalsh(clipped)
        assert eigs.min() >= -1e-14  # numerically zero

    def test_tol_parameter_respected(self):
        """With tol=1.0, eigenvalues below 1.0 are raised to 1.0."""
        M = np.diag([0.5, 2.0])
        clipped = SigmaObject._clip_psd(M, 1.0)
        eigs = np.linalg.eigvalsh(clipped)
        assert eigs.min() >= 1.0 - 1e-12


# ===========================================================================
# var — quadratic form w' Sigma w
# ===========================================================================

class TestVar:
    def test_identity_R_known_value(self):
        """sigma=[3,4], R=I, w=[0.6,0.4]: var = (0.6*3)^2 + (0.4*4)^2 = 5.8."""
        sigma = np.array([3.0, 4.0])
        R = np.eye(2)
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        w = np.array([0.6, 0.4])
        np.testing.assert_allclose(S.var(w), 5.8, rtol=1e-12)

    def test_returns_float(self):
        S = _make_object(n=2)
        result = S.var(np.array([0.5, 0.5]))
        assert isinstance(result, float)

    def test_nonneg_for_psd_sigma(self):
        S = _make_object(n=3, rho=0.4)
        assert S.var(np.array([0.4, 0.3, 0.3])) >= 0.0

    def test_zero_weight_component_does_not_contribute(self):
        sigma = np.array([2.0, 1.0, 3.0])
        R = _equicorr(3, 0.3)
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b", "c"])
        w_full = np.array([0.5, 0.5, 0.0])
        w_sub = np.array([0.5, 0.5])
        S_sub = SigmaObject(sigma=sigma[:2], R=R[:2, :2], labels=["a", "b"])
        np.testing.assert_allclose(S.var(w_full), S_sub.var(w_sub), rtol=1e-12)

    def test_accepts_list_weights(self):
        S = _make_object(n=2)
        result = S.var([0.5, 0.5])
        assert isinstance(result, float)

    def test_correlated_var_exceeds_uncorrelated(self):
        sigma = np.array([1.0, 1.0])
        w = np.array([0.5, 0.5])
        S_uncorr = SigmaObject(sigma=sigma, R=np.eye(2), labels=["a", "b"])
        S_corr = SigmaObject(sigma=sigma, R=_equicorr(2, 0.8), labels=["a", "b"])
        assert S_corr.var(w) > S_uncorr.var(w)


# ===========================================================================
# var_no_cov — diagonal-only variance
# ===========================================================================

class TestVarNoCov:
    def test_identity_R_equals_var(self):
        """With identity R, off-diagonal is zero, so var_no_cov == var."""
        sigma = np.array([2.0, 3.0])
        R = np.eye(2)
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        w = np.array([0.3, 0.7])
        np.testing.assert_allclose(S.var_no_cov(w), S.var(w), rtol=1e-12)

    def test_known_value(self):
        """sigma=[2,1], w=[0.5,0.5]: var_no_cov = (0.5*2)^2 + (0.5*1)^2 = 1.25."""
        sigma = np.array([2.0, 1.0])
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        np.testing.assert_allclose(S.var_no_cov(np.array([0.5, 0.5])), 1.25, rtol=1e-12)

    def test_returns_float(self):
        S = _make_object(n=2)
        assert isinstance(S.var_no_cov(np.array([0.5, 0.5])), float)

    def test_nonneg(self):
        S = _make_object(n=3, rho=0.5)
        assert S.var_no_cov(np.array([0.4, 0.3, 0.3])) >= 0.0

    def test_scales_quadratically_with_sigma(self):
        sigma_base = np.array([1.0, 1.0])
        sigma_2x = np.array([2.0, 2.0])
        R = np.eye(2)
        w = np.array([0.5, 0.5])
        S1 = SigmaObject(sigma=sigma_base, R=R, labels=["a", "b"])
        S2 = SigmaObject(sigma=sigma_2x, R=R, labels=["a", "b"])
        np.testing.assert_allclose(S2.var_no_cov(w), 4.0 * S1.var_no_cov(w), rtol=1e-12)


# ===========================================================================
# cov_premium_var — with zero-denominator branch
# ===========================================================================

class TestCovPremiumVar:
    def test_identity_R_zero_premium(self):
        """No correlation means zero covariance premium."""
        sigma = np.array([1.0, 2.0])
        R = np.eye(2)
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        np.testing.assert_allclose(S.cov_premium_var(np.array([0.5, 0.5])), 0.0, atol=1e-14)

    def test_known_positive_value(self):
        """sigma=[2,1], R=[[1,.5],[.5,1]], w=[0.5,0.5]:
        v=1.75, v_diag=1.25, premium=(1.75-1.25)/1.25=0.4."""
        sigma = np.array([2.0, 1.0])
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        np.testing.assert_allclose(
            S.cov_premium_var(np.array([0.5, 0.5])), 0.4, rtol=1e-12
        )

    def test_zero_denominator_returns_zero(self):
        """When sigma=0 (v_diag=0), must return 0.0 without division error."""
        S = SigmaObject(sigma=np.array([0.0, 0.0]), R=np.eye(2), labels=["a", "b"])
        result = S.cov_premium_var(np.array([0.5, 0.5]))
        assert result == 0.0

    def test_zero_weight_on_zero_sigma_returns_zero(self):
        """w=0 means v_diag=0, should return 0.0."""
        sigma = np.array([1.0, 2.0])
        R = np.eye(2)
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        result = S.cov_premium_var(np.array([0.0, 0.0]))
        assert result == 0.0

    def test_premium_increases_with_correlation(self):
        sigma = np.array([1.0, 1.0])
        w = np.array([0.5, 0.5])
        S_low = SigmaObject(sigma=sigma, R=_equicorr(2, 0.2), labels=["a", "b"])
        S_high = SigmaObject(sigma=sigma, R=_equicorr(2, 0.8), labels=["a", "b"])
        assert S_high.cov_premium_var(w) > S_low.cov_premium_var(w)

    def test_returns_float(self):
        S = _make_object(n=2, rho=0.3)
        assert isinstance(S.cov_premium_var(np.array([0.5, 0.5])), float)


# ===========================================================================
# mean_corr — including n=1 edge case (off.size == 0)
# ===========================================================================

class TestMeanCorr:
    def test_n1_returns_zero(self):
        """With only 1 component there are no off-diagonal entries; returns 0.0."""
        S = SigmaObject(sigma=np.array([2.0]), R=np.array([[1.0]]), labels=["a"])
        assert S.mean_corr() == 0.0

    def test_n2_known_value(self):
        """n=2: single off-diagonal entry rho=0.5."""
        sigma = np.array([1.0, 1.0])
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        np.testing.assert_allclose(S.mean_corr(), 0.5, atol=1e-14)

    def test_n3_equicorr_known_value(self):
        """Equicorrelation rho=0.4 -> mean_corr == 0.4."""
        S = SigmaObject(sigma=np.ones(3), R=_equicorr(3, 0.4), labels=["a","b","c"])
        np.testing.assert_allclose(S.mean_corr(), 0.4, atol=1e-13)

    def test_identity_R_mean_corr_zero(self):
        S = SigmaObject(sigma=np.ones(4), R=np.eye(4), labels=list("abcd"))
        np.testing.assert_allclose(S.mean_corr(), 0.0, atol=1e-14)

    def test_returns_float(self):
        S = _make_object(n=3, rho=0.3)
        assert isinstance(S.mean_corr(), float)

    def test_mean_corr_uses_R_not_Sigma(self):
        """mean_corr is the mean off-diagonal of the correlation matrix R,
        not of Sigma, so sigma scaling should not affect it."""
        R = np.array([[1.0, 0.3], [0.3, 1.0]])
        S1 = SigmaObject(sigma=np.array([1.0, 1.0]), R=R, labels=["a","b"])
        S2 = SigmaObject(sigma=np.array([10.0, 0.1]), R=R, labels=["a","b"])
        np.testing.assert_allclose(S1.mean_corr(), S2.mean_corr(), atol=1e-14)


# ===========================================================================
# diag_contrib — shape, dtype, conservation
# ===========================================================================

class TestDiagContrib:
    def test_shape_matches_n(self):
        for n in [1, 2, 4]:
            S = _make_object(n=n, rho=0.2)
            w = np.ones(n) / n
            dc = S.diag_contrib(w)
            assert dc.shape == (n,)

    def test_known_value(self):
        """sigma=[2,1], w=[0.5,0.5]: diag_contrib = [(0.5*2)^2, (0.5*1)^2] = [1.0, 0.25]."""
        sigma = np.array([2.0, 1.0])
        R = np.array([[1.0, 0.5], [0.5, 1.0]])
        S = SigmaObject(sigma=sigma, R=R, labels=["a", "b"])
        dc = S.diag_contrib(np.array([0.5, 0.5]))
        np.testing.assert_allclose(dc, [1.0, 0.25], rtol=1e-12)

    def test_sums_to_var_no_cov(self):
        """diag_contrib sums to var_no_cov (conservation)."""
        sigma = np.array([1.0, 2.0, 0.5])
        R = _equicorr(3, 0.3)
        S = SigmaObject(sigma=sigma, R=R, labels=["a","b","c"])
        w = np.array([0.4, 0.3, 0.3])
        dc = S.diag_contrib(w)
        np.testing.assert_allclose(dc.sum(), S.var_no_cov(w), rtol=1e-12)

    def test_nonneg(self):
        S = _make_object(n=3, rho=0.4)
        dc = S.diag_contrib(np.array([0.4, 0.3, 0.3]))
        assert dc.min() >= 0.0

    def test_accepts_list_weights(self):
        S = _make_object(n=2)
        dc = S.diag_contrib([0.5, 0.5])
        assert dc.shape == (2,)

    def test_scales_quadratically_with_weight(self):
        """Doubling one weight quadruples its diag_contrib."""
        sigma = np.array([1.0, 1.0])
        R = np.eye(2)
        S = SigmaObject(sigma=sigma, R=R, labels=["a","b"])
        w1 = np.array([0.5, 0.5])
        w2 = np.array([1.0, 0.5])  # component 0 doubled
        dc1 = S.diag_contrib(w1)
        dc2 = S.diag_contrib(w2)
        np.testing.assert_allclose(dc2[0], 4.0 * dc1[0], rtol=1e-12)

    def test_zero_weight_gives_zero_contrib(self):
        sigma = np.array([2.0, 3.0])
        R = np.eye(2)
        S = SigmaObject(sigma=sigma, R=R, labels=["a","b"])
        dc = S.diag_contrib(np.array([0.0, 1.0]))
        assert dc[0] == 0.0


# ===========================================================================
# End-to-end and cross-method consistency
# ===========================================================================

class TestCrossMethodConsistency:
    def test_var_decomposes_into_diag_plus_cov_premium(self):
        """var(w) == var_no_cov(w) + cov_premium_var(w) * var_no_cov(w)
        i.e. var(w) == var_no_cov(w) * (1 + cov_premium_var(w))."""
        sigma = np.array([1.0, 2.0, 0.5])
        R = _equicorr(3, 0.4)
        S = SigmaObject(sigma=sigma, R=R, labels=["a","b","c"])
        w = np.array([0.4, 0.3, 0.3])
        v = S.var(w)
        vd = S.var_no_cov(w)
        prem = S.cov_premium_var(w)
        np.testing.assert_allclose(v, vd * (1 + prem), rtol=1e-12)

    def test_diag_contrib_sum_plus_cov_equals_var(self):
        """diag_contrib.sum() + (var - var_no_cov) == var."""
        sigma = np.array([1.0, 2.0])
        R = np.array([[1.0, 0.4], [0.4, 1.0]])
        S = SigmaObject(sigma=sigma, R=R, labels=["a","b"])
        w = np.array([0.6, 0.4])
        v = S.var(w)
        dc_sum = S.diag_contrib(w).sum()
        cov_abs = v - S.var_no_cov(w)
        np.testing.assert_allclose(dc_sum + cov_abs, v, rtol=1e-12)

    def test_reproducible_across_two_instantiations(self):
        """Same inputs produce identical Sigma on separate instantiations."""
        sigma = np.array([1.5, 0.8, 2.0])
        R = _equicorr(3, 0.25)
        labels = ["x", "y", "z"]
        S1 = SigmaObject(sigma=sigma.copy(), R=R.copy(), labels=labels)
        S2 = SigmaObject(sigma=sigma.copy(), R=R.copy(), labels=labels)
        np.testing.assert_array_equal(S1.Sigma, S2.Sigma)

    def test_n1_var_equals_sigma_squared_times_w_squared(self):
        """Scalar case: var = sigma^2 * w^2."""
        sigma = np.array([3.0])
        S = SigmaObject(sigma=sigma, R=np.array([[1.0]]), labels=["x"])
        w = np.array([0.7])
        np.testing.assert_allclose(S.var(w), (3.0 * 0.7) ** 2, rtol=1e-12)
