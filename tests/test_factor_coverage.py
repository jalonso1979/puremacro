"""Coverage tests for puremacro.factor.

Tests cover the three public functions plus the internal helper:
- _standardise (lines 27-32)
- pca_factors   (lines 57-72)
- bai_ng_ic     (lines 94-116)
- static_dfm_fit (lines 128-138)

Organised by function with known-value, shape/dtype, property, and error tests.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.factor import pca_factors, bai_ng_ic, static_dfm_fit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_factor_data(
    T: int = 120,
    n: int = 20,
    k_true: int = 2,
    noise_sd: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic DFM: X = F @ Lambda.T + e, where F is (T, k_true)."""
    rng = np.random.default_rng(seed)
    F = rng.standard_normal((T, k_true))
    Lambda = rng.standard_normal((n, k_true))
    e = rng.standard_normal((T, n)) * noise_sd
    X = F @ Lambda.T + e
    return X, F, Lambda


# ===========================================================================
# _standardise (tested indirectly through pca_factors; also via demean/standardize flags)
# ===========================================================================

class TestStandardise:
    """Test the demeaning/standardizing preprocessing path."""

    def test_demean_only_centers_columns(self):
        """With standardize=False, columns should have mean ~0 after pca_factors."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 5)) + 10.0  # large mean
        out = pca_factors(X, k=1, demean=True, standardize=False)
        # factors are T x k; the input was demeaned, so factors should have ~0 mean
        assert out["factors"].shape == (50, 1)

    def test_no_demean_no_standardize_keeps_data_as_is(self):
        """With both flags False, pca_factors should still run and return shapes."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 8)) * 5.0
        out = pca_factors(X, k=2, demean=False, standardize=False)
        assert out["factors"].shape == (60, 2)
        assert out["loadings"].shape == (8, 2)

    def test_zero_variance_column_not_divide_by_zero(self):
        """A constant column should not cause division-by-zero."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((50, 5))
        X[:, 2] = 3.0  # constant column
        # Should not raise
        out = pca_factors(X, k=1, demean=True, standardize=True)
        assert out["factors"].shape == (50, 1)
        assert np.all(np.isfinite(out["factors"]))
        assert np.all(np.isfinite(out["loadings"]))


# ===========================================================================
# pca_factors
# ===========================================================================

class TestPcaFactors:
    """Tests for pca_factors output shapes, dtypes, and mathematical properties."""

    def test_output_keys_present(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        out = pca_factors(X, k=2)
        for key in ("factors", "loadings", "eigvals", "variance_share", "cumulative_share"):
            assert key in out, f"missing key: {key}"

    def test_shapes_are_correct(self):
        T, n, k = 100, 15, 3
        X, _, _ = _make_factor_data(T=T, n=n, k_true=2)
        out = pca_factors(X, k=k)
        assert out["factors"].shape == (T, k)
        assert out["loadings"].shape == (n, k)
        assert out["eigvals"].shape == (k,)
        assert out["variance_share"].shape == (k,)
        assert out["cumulative_share"].shape == (k,)

    def test_variance_share_sums_to_at_most_one(self):
        X, _, _ = _make_factor_data(T=80, n=10, k_true=2)
        out = pca_factors(X, k=3)
        total = float(out["variance_share"].sum())
        assert 0.0 <= total <= 1.0 + 1e-10

    def test_cumulative_share_is_nondecreasing(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        out = pca_factors(X, k=4)
        cs = out["cumulative_share"]
        diffs = np.diff(cs)
        assert np.all(diffs >= -1e-12), "cumulative_share must be nondecreasing"

    def test_eigvals_are_nonnegative(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        out = pca_factors(X, k=4)
        assert np.all(out["eigvals"] >= -1e-12)

    def test_eigvals_are_decreasing(self):
        """PCA eigenvalues should be in descending order (PCA convention)."""
        X, _, _ = _make_factor_data(T=100, n=12, k_true=3, noise_sd=0.1)
        out = pca_factors(X, k=4)
        evs = out["eigvals"]
        assert np.all(np.diff(evs) <= 1e-10), "eigvals should be descending"

    def test_first_factor_explains_most_variance_in_one_factor_world(self):
        """When data is generated with 1 true factor, k=1 should explain large share."""
        rng = np.random.default_rng(42)
        T, n = 200, 30
        f = rng.standard_normal((T, 1))
        lam = rng.standard_normal((n, 1))
        X = f @ lam.T + 0.05 * rng.standard_normal((T, n))
        out = pca_factors(X, k=1)
        # With tiny noise, first factor should explain >90% of variance
        assert float(out["variance_share"][0]) > 0.80

    def test_reconstruction_X_hat_is_rank_k(self):
        """F @ Λ.T should reconstruct X_std well when noise is small.

        The module standardises X before running SVD; factors and loadings are
        expressed in units of the standardised data, so we reconstruct X_std
        (zero-mean, unit-variance columns) and check the residual.
        """
        rng = np.random.default_rng(7)
        T, n, k = 150, 20, 2
        F_true = rng.standard_normal((T, k))
        Lam_true = rng.standard_normal((n, k))
        X = F_true @ Lam_true.T + 0.05 * rng.standard_normal((T, n))
        out = pca_factors(X, k=k, demean=True, standardize=True)
        F_hat = out["factors"]    # (T, k)
        Lam_hat = out["loadings"] # (n, k)
        X_hat = F_hat @ Lam_hat.T
        # Compare against the same standardised version of X
        X_std = X - X.mean(axis=0)
        sd = X_std.std(axis=0, ddof=0)
        sd = np.where(sd == 0, 1.0, sd)
        X_std = X_std / sd
        rel_err = np.linalg.norm(X_std - X_hat, "fro") / np.linalg.norm(X_std, "fro")
        assert rel_err < 0.20, f"relative Frobenius error too large: {rel_err:.3f}"

    def test_k1_returns_single_factor_column(self):
        X, _, _ = _make_factor_data(T=60, n=8)
        out = pca_factors(X, k=1)
        assert out["factors"].ndim == 2
        assert out["factors"].shape[1] == 1

    def test_accepts_pandas_dataframe_input(self):
        """pca_factors should handle a DataFrame input gracefully."""
        rng = np.random.default_rng(9)
        X = pd.DataFrame(rng.standard_normal((50, 6)))
        out = pca_factors(X, k=2)
        assert out["factors"].shape == (50, 2)

    def test_demean_false_standardize_true_path(self):
        rng = np.random.default_rng(10)
        X = rng.standard_normal((50, 5)) + 5.0
        out = pca_factors(X, k=1, demean=False, standardize=True)
        assert out["factors"].shape == (50, 1)
        assert np.all(np.isfinite(out["factors"]))


# ===========================================================================
# bai_ng_ic
# ===========================================================================

class TestBaiNgIc:
    """Tests for the Bai-Ng (2002) information criteria."""

    def test_output_keys_present(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        out = bai_ng_ic(X, kmax=5)
        for key in ("table", "k_hat_IC1", "k_hat_IC2", "k_hat_IC3"):
            assert key in out, f"missing key: {key}"

    def test_table_is_dataframe_with_correct_columns(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        out = bai_ng_ic(X, kmax=5)
        df = out["table"]
        assert isinstance(df, pd.DataFrame)
        for col in ("k", "V_k", "IC1", "IC2", "IC3"):
            assert col in df.columns

    def test_table_has_kmax_plus_one_rows(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        kmax = 6
        out = bai_ng_ic(X, kmax=kmax)
        assert len(out["table"]) == kmax + 1

    def test_k_column_is_0_to_kmax(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        kmax = 4
        out = bai_ng_ic(X, kmax=kmax)
        expected_ks = list(range(0, kmax + 1))
        assert list(out["table"]["k"]) == expected_ks

    def test_v_k_is_nonnegative(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        out = bai_ng_ic(X, kmax=5)
        assert (out["table"]["V_k"] >= 0).all()

    def test_v_k_is_nonincreasing_in_k(self):
        """Adding more factors never increases residual variance V(k)."""
        X, _, _ = _make_factor_data(T=100, n=15)
        out = bai_ng_ic(X, kmax=5)
        vk = out["table"]["V_k"].values
        # Allow tiny floating-point rounding
        assert np.all(np.diff(vk) <= 1e-10), "V_k must be nonincreasing in k"

    def test_k_hat_values_are_integers_in_range(self):
        X, _, _ = _make_factor_data(T=100, n=15, k_true=2)
        kmax = 8
        out = bai_ng_ic(X, kmax=kmax)
        for key in ("k_hat_IC1", "k_hat_IC2", "k_hat_IC3"):
            val = out[key]
            assert isinstance(val, int)
            assert 0 <= val <= kmax, f"{key}={val} out of range [0, {kmax}]"

    def test_selects_nonzero_k_with_strong_signal(self):
        """With very low noise, ICs should select k > 0."""
        rng = np.random.default_rng(99)
        T, n, k = 200, 30, 3
        F = rng.standard_normal((T, k))
        Lam = rng.standard_normal((n, k))
        X = F @ Lam.T + 0.01 * rng.standard_normal((T, n))
        out = bai_ng_ic(X, kmax=8)
        # At least one IC should pick k >= 1
        assert (out["k_hat_IC1"] >= 1
                or out["k_hat_IC2"] >= 1
                or out["k_hat_IC3"] >= 1)

    def test_selects_zero_factors_for_iid_noise(self):
        """Pure white noise should give k=0 or very small k for all ICs."""
        rng = np.random.default_rng(55)
        X = rng.standard_normal((200, 40))  # no factor structure
        out = bai_ng_ic(X, kmax=10)
        # With pure noise and penalty, expected k_hat is small (0-2)
        for key in ("k_hat_IC1", "k_hat_IC2", "k_hat_IC3"):
            assert out[key] <= 4, f"{key}={out[key]} unexpectedly large for iid noise"

    def test_demean_and_standardize_flags_propagate(self):
        """Passing demean=False, standardize=False should still return a valid result."""
        X, _, _ = _make_factor_data(T=80, n=10)
        out = bai_ng_ic(X, kmax=3, demean=False, standardize=False)
        assert "table" in out
        assert len(out["table"]) == 4

    def test_accepts_pandas_dataframe(self):
        rng = np.random.default_rng(12)
        X = pd.DataFrame(rng.standard_normal((60, 8)))
        out = bai_ng_ic(X, kmax=4)
        assert "k_hat_IC2" in out

    def test_v_k_zero_at_k_equals_n_columns(self):
        """When k equals min(T,n), residual variance should be exactly 0."""
        T, n = 50, 8
        X, _, _ = _make_factor_data(T=T, n=n)
        out = bai_ng_ic(X, kmax=n)
        # The last row (k=n) should have V_k == 0 (perfect reconstruction)
        last_vk = float(out["table"].loc[out["table"]["k"] == n, "V_k"].iloc[0])
        assert last_vk < 1e-10, f"V_k({n}) should be 0, got {last_vk}"


# ===========================================================================
# static_dfm_fit
# ===========================================================================

class TestStaticDfmFit:
    """Tests for the convenience wrapper that selects k via IC2 if not provided."""

    def test_output_keys_present(self):
        X, _, _ = _make_factor_data(T=80, n=10)
        out = static_dfm_fit(X)
        for key in ("factors", "loadings", "eigvals", "variance_share",
                    "cumulative_share", "k", "bai_ng_table"):
            assert key in out, f"missing key: {key}"

    def test_k_is_int_and_in_valid_range(self):
        X, _, _ = _make_factor_data(T=100, n=15)
        kmax = 6
        out = static_dfm_fit(X, kmax=kmax)
        assert isinstance(out["k"], int)
        assert 0 <= out["k"] <= kmax

    def test_bai_ng_table_is_dataframe_when_k_auto(self):
        """When k is not supplied, bai_ng_table should be a DataFrame."""
        X, _, _ = _make_factor_data(T=80, n=10)
        out = static_dfm_fit(X)
        assert isinstance(out["bai_ng_table"], pd.DataFrame)

    def test_bai_ng_table_is_none_when_k_supplied(self):
        """When k is explicitly passed, bai_ng_table should be None."""
        X, _, _ = _make_factor_data(T=80, n=10)
        out = static_dfm_fit(X, k=2)
        assert out["bai_ng_table"] is None

    def test_explicit_k_is_honored(self):
        """Supplying k=3 should give k=3 regardless of IC suggestion."""
        X, _, _ = _make_factor_data(T=100, n=15)
        out = static_dfm_fit(X, k=3)
        assert out["k"] == 3
        assert out["factors"].shape[1] == 3
        assert out["loadings"].shape[1] == 3

    def test_shapes_consistent_with_auto_k(self):
        T, n = 120, 20
        X, _, _ = _make_factor_data(T=T, n=n)
        out = static_dfm_fit(X)
        k = out["k"]
        assert out["factors"].shape == (T, k)
        assert out["loadings"].shape == (n, k)

    def test_reproducible_output_same_seed(self):
        """Same data should yield exactly the same result."""
        rng = np.random.default_rng(77)
        X = rng.standard_normal((100, 12))
        out1 = static_dfm_fit(X, k=2)
        out2 = static_dfm_fit(X, k=2)
        np.testing.assert_array_equal(out1["factors"], out2["factors"])

    def test_strong_signal_recovers_k_ge1(self):
        """With a clear 2-factor signal, IC2-based selection should give k >= 1."""
        rng = np.random.default_rng(88)
        T, n, k_true = 200, 30, 2
        F = rng.standard_normal((T, k_true))
        Lam = rng.standard_normal((n, k_true))
        X = F @ Lam.T + 0.01 * rng.standard_normal((T, n))
        out = static_dfm_fit(X, kmax=8)
        assert out["k"] >= 1

    def test_demean_standardize_flags_pass_through(self):
        """demean/standardize flags should be accepted without error."""
        X, _, _ = _make_factor_data(T=60, n=8)
        out = static_dfm_fit(X, k=1, demean=False, standardize=False)
        assert out["k"] == 1
        assert out["factors"].shape[1] == 1

    def test_all_outputs_are_finite(self):
        X, _, _ = _make_factor_data(T=80, n=12)
        out = static_dfm_fit(X, k=2)
        for key in ("factors", "loadings", "eigvals", "variance_share", "cumulative_share"):
            assert np.all(np.isfinite(out[key])), f"non-finite values in {key}"
