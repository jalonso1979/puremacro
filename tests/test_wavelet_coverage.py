"""Tests for puremacro.wavelet — discrete and maximal-overlap Haar DWT.

Strategy:
- Known-value: build small arrays with analytically known Haar DWT coefficients
  or known energy; assert the implementation matches.
- Properties: output shapes, energy conservation, IDWT roundtrip, MODWT
  time-alignment (all coefficient arrays have length T), period-band structure,
  variance non-negativity, shares in [0,1], coherence in [-1,1].
- Known-DGP: synthetic sinusoids at specific periods localize power in the
  expected frequency band; near-identical signals yield coherence close to +1;
  negated signals yield coherence close to -1.
- Edge cases: constant input, zero-mean check, length-mismatch raises ValueError,
  frozen dataclasses raise on assignment.
- Summary strings: .summary() returns a str for both result dataclasses.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.wavelet import (
    WaveletCoherenceResult,
    WaveletVarianceResult,
    dwt_haar,
    idwt_haar,
    modwt_haar,
    wavelet_coherence,
    wavelet_variance,
)

RNG = np.random.default_rng(2025)


# ===========================================================================
# 1. dwt_haar — known-value and structural tests
# ===========================================================================

class TestDwtHaar:
    def test_level1_cA_known_value(self):
        """cA[j] = (x[2j] + x[2j+1]) / sqrt(2)."""
        x = np.array([1.0, 3.0, 5.0, 7.0])
        coeffs = dwt_haar(x, level=1)
        # coeffs = [cA_1, cD_1]
        expected_cA = np.array([(1 + 3) / np.sqrt(2), (5 + 7) / np.sqrt(2)])
        np.testing.assert_allclose(coeffs[0], expected_cA, rtol=1e-14)

    def test_level1_cD_known_value(self):
        """cD[j] = (x[2j] - x[2j+1]) / sqrt(2)."""
        x = np.array([1.0, 3.0, 5.0, 7.0])
        coeffs = dwt_haar(x, level=1)
        expected_cD = np.array([(1 - 3) / np.sqrt(2), (5 - 7) / np.sqrt(2)])
        np.testing.assert_allclose(coeffs[1], expected_cD, rtol=1e-14)

    def test_level2_approximation_known_value(self):
        """Level-2 approximation: two stages of averaging."""
        x = np.array([1.0, 3.0, 5.0, 7.0])
        coeffs = dwt_haar(x, level=2)
        # cA_1 = [2*sqrt(2), 6*sqrt(2)]
        # cA_2 = [(2*sqrt(2) + 6*sqrt(2)) / sqrt(2)] = [8]
        # cD_2 = [(2*sqrt(2) - 6*sqrt(2)) / sqrt(2)] = [-4]
        # cD_1 = [(-1/sqrt(2))*2, ...] = [-sqrt(2), -sqrt(2)]
        expected_cA2 = np.array([8.0])
        np.testing.assert_allclose(coeffs[0], expected_cA2, rtol=1e-14)

    def test_energy_conservation(self):
        """Parseval: sum of squared coefficients == sum of squares of input."""
        for T in [8, 16, 32]:
            rng = np.random.default_rng(T)
            x = rng.standard_normal(T)
            level = int(np.floor(np.log2(T))) - 1
            coeffs = dwt_haar(x, level=level)
            energy_x = float(np.sum(x ** 2))
            energy_coeffs = float(sum(np.sum(c ** 2) for c in coeffs))
            assert abs(energy_x - energy_coeffs) < 1e-10, (
                f"T={T}: energy_x={energy_x:.6f} != energy_coeffs={energy_coeffs:.6f}"
            )

    def test_output_length(self):
        """Output is a list of level+1 arrays (cA_J + J detail arrays)."""
        x = RNG.standard_normal(32)
        for level in [1, 2, 3, 4]:
            coeffs = dwt_haar(x, level=level)
            assert len(coeffs) == level + 1, (
                f"level={level}: expected {level+1} arrays, got {len(coeffs)}"
            )

    def test_approx_length_halved_each_level(self):
        """Approximation length halves at every level."""
        x = RNG.standard_normal(32)
        coeffs = dwt_haar(x, level=4)
        # cA_J is coeffs[0], length should be T / 2^level = 32/16 = 2
        assert len(coeffs[0]) == 2

    def test_detail_coeff_lengths(self):
        """Level-j detail has length T/2^j."""
        T = 32
        x = RNG.standard_normal(T)
        coeffs = dwt_haar(x, level=4)
        # coeffs[0]=cA_4, coeffs[1]=cD_4, ..., coeffs[4]=cD_1
        # cD_j has length T/2^j
        # j=4: length 32/16=2; j=1: length 32/2=16
        assert len(coeffs[1]) == T // (2 ** 4)  # cD_4: T/16=2
        assert len(coeffs[4]) == T // (2 ** 1)  # cD_1: T/2=16

    def test_default_level(self):
        """Default level is floor(log2(T)) - 1."""
        for T in [16, 32, 64]:
            x = RNG.standard_normal(T)
            coeffs = dwt_haar(x)
            expected_level = int(np.floor(np.log2(T))) - 1
            assert len(coeffs) - 1 == expected_level, (
                f"T={T}: expected level={expected_level}, got {len(coeffs)-1}"
            )

    def test_list_input_accepted(self):
        """Python lists are cast to ndarray without error."""
        x_list = [1.0, -1.0, 2.0, -2.0]
        coeffs = dwt_haar(x_list, level=1)
        assert isinstance(coeffs[0], np.ndarray)

    def test_returns_float_arrays(self):
        """All output arrays have float dtype."""
        x = np.arange(8, dtype=int)
        coeffs = dwt_haar(x, level=2)
        for c in coeffs:
            assert np.issubdtype(c.dtype, np.floating)

    def test_constant_series_details_zero(self):
        """Constant input: all detail coefficients are zero."""
        x = np.full(16, 3.14)
        coeffs = dwt_haar(x, level=3)
        # Detail coefficients (indices 1..level) must be zero
        for cD in coeffs[1:]:
            np.testing.assert_allclose(cD, 0.0, atol=1e-14)


# ===========================================================================
# 2. idwt_haar — roundtrip and exact reconstruction
# ===========================================================================

class TestIdwtHaar:
    def test_roundtrip_power_of_two(self):
        """DWT followed by IDWT recovers the original signal."""
        for T in [8, 16, 32]:
            rng = np.random.default_rng(T + 100)
            x = rng.standard_normal(T)
            level = int(np.floor(np.log2(T))) - 1
            coeffs = dwt_haar(x, level=level)
            x_rec = idwt_haar(coeffs)
            np.testing.assert_allclose(
                x_rec, x[:len(x_rec)], atol=1e-13,
                err_msg=f"T={T}: IDWT reconstruction failed"
            )

    def test_level1_known_inverse(self):
        """Level-1 IDWT: out[2j] = (cA[j]+cD[j])/sqrt(2), out[2j+1] = (cA[j]-cD[j])/sqrt(2)."""
        cA = np.array([2.0 * np.sqrt(2), 6.0 * np.sqrt(2)])
        cD = np.array([-np.sqrt(2), -np.sqrt(2)])
        x_rec = idwt_haar([cA, cD])
        expected = np.array([1.0, 3.0, 5.0, 7.0])
        np.testing.assert_allclose(x_rec, expected, atol=1e-13)

    def test_output_length_is_2n(self):
        """IDWT of level-1 reconstruction doubles the approximation length."""
        cA = np.array([1.0, 2.0, 3.0, 4.0])
        cD = np.zeros(4)
        x_rec = idwt_haar([cA, cD])
        assert len(x_rec) == 8

    def test_single_level_roundtrip(self):
        """Level=1 DWT -> IDWT must recover first 2*n samples."""
        x = np.array([4.0, 2.0, 6.0, 8.0])
        coeffs = dwt_haar(x, level=1)
        x_rec = idwt_haar(coeffs)
        np.testing.assert_allclose(x_rec, x, atol=1e-13)


# ===========================================================================
# 3. modwt_haar — time-alignment and structural contracts
# ===========================================================================

class TestModwtHaar:
    def test_all_coefficients_have_length_T(self):
        """MODWT is undecimated: every output array has the same length as x."""
        for T in [16, 32, 64]:
            rng = np.random.default_rng(T + 200)
            x = rng.standard_normal(T)
            coeffs = modwt_haar(x, level=3)
            for j, c in enumerate(coeffs):
                assert len(c) == T, (
                    f"T={T}, coeff[{j}] has length {len(c)} (expected {T})"
                )

    def test_output_list_length(self):
        """Returns level+1 arrays (cA_J + J detail arrays)."""
        x = RNG.standard_normal(32)
        for level in [1, 2, 3, 4]:
            coeffs = modwt_haar(x, level=level)
            assert len(coeffs) == level + 1

    def test_constant_input_detail_near_zero(self):
        """Constant signal: all detail coefficients should be ~0."""
        x = np.full(32, 7.0)
        coeffs = modwt_haar(x, level=3)
        for cD in coeffs[1:]:
            np.testing.assert_allclose(
                cD, 0.0, atol=1e-13,
                err_msg="MODWT detail of constant signal should be 0"
            )

    def test_list_input_accepted(self):
        """Python list input is silently cast to ndarray."""
        x = list(range(16))
        coeffs = modwt_haar(x, level=2)
        for c in coeffs:
            assert isinstance(c, np.ndarray)

    def test_returns_float_arrays(self):
        x = np.arange(16, dtype=int)
        coeffs = modwt_haar(x, level=2)
        for c in coeffs:
            assert np.issubdtype(c.dtype, np.floating)

    def test_short_signal_filter_longer_than_signal(self):
        """At high levels, the dilated filter can exceed signal length.
        This exercises the L > n branch in _convolve_circular (lines 98-99)."""
        # At level=4, gap=8, filter length = (2-1)*8+1 = 9, which exceeds n=3
        x = np.array([1.0, 2.0, 3.0])
        coeffs = modwt_haar(x, level=4)
        # All coefficients must still have length T=3
        for c in coeffs:
            assert len(c) == 3
            assert np.isfinite(c).all()


# ===========================================================================
# 4. wavelet_variance — properties and DGP localization
# ===========================================================================

class TestWaveletVariance:
    def test_output_type_is_result(self):
        x = RNG.standard_normal(64)
        result = wavelet_variance(x, level=4)
        assert isinstance(result, WaveletVarianceResult)

    def test_variance_per_scale_shape(self):
        """variance_per_scale has shape (level,)."""
        for level in [2, 3, 4]:
            x = RNG.standard_normal(64)
            result = wavelet_variance(x, level=level)
            assert result.variance_per_scale.shape == (level,)

    def test_share_shape_matches_level(self):
        x = RNG.standard_normal(64)
        result = wavelet_variance(x, level=3)
        assert result.share.shape == (3,)

    def test_variance_per_scale_nonnegative(self):
        """Variance estimates must be non-negative."""
        x = RNG.standard_normal(128)
        result = wavelet_variance(x, level=5)
        assert (result.variance_per_scale >= 0).all()

    def test_share_nonnegative(self):
        x = RNG.standard_normal(64)
        result = wavelet_variance(x, level=4)
        assert (result.share >= 0).all()

    def test_share_sum_at_most_one(self):
        """Detail coefficients do not capture the approximation residual,
        so share can sum to < 1 but must not exceed 1."""
        x = RNG.standard_normal(64)
        result = wavelet_variance(x, level=4)
        assert result.share.sum() <= 1.0 + 1e-12

    def test_total_is_mean_squared_demeaned(self):
        """total = mean((x - mean(x))^2)."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal(64) + 5.0  # non-zero mean
        result = wavelet_variance(x, level=3)
        expected_total = float(((x - x.mean()) ** 2).mean())
        assert abs(result.total - expected_total) < 1e-13

    def test_constant_input_total_is_zero(self):
        """Constant signal has zero variance."""
        x = np.full(64, 2.5)
        result = wavelet_variance(x, level=3)
        assert result.total == pytest.approx(0.0, abs=1e-13)

    def test_constant_input_share_is_zero(self):
        """With zero total variance, share vector must be all zeros."""
        x = np.full(64, 2.5)
        result = wavelet_variance(x, level=3)
        np.testing.assert_array_equal(result.share, np.zeros(3))

    def test_period_bands_are_correct(self):
        """period_bands[j] = (2^{j+1}, 2^{j+2}) for j=0,1,...,level-1."""
        x = RNG.standard_normal(64)
        level = 4
        result = wavelet_variance(x, level=level)
        for j in range(level):
            expected_lo = 2 ** (j + 1)
            expected_hi = 2 ** (j + 2)
            assert result.period_bands[j] == (expected_lo, expected_hi), (
                f"j={j}: got {result.period_bands[j]}, expected ({expected_lo},{expected_hi})"
            )

    def test_period_bands_ordered_increasing(self):
        """Period bands are ordered from finest (j=1) to coarsest scale."""
        x = RNG.standard_normal(64)
        result = wavelet_variance(x, level=4)
        los = [b[0] for b in result.period_bands]
        assert los == sorted(los), "period_bands should be ordered finest to coarsest"

    def test_n_bands_matches_level(self):
        x = RNG.standard_normal(64)
        for level in [2, 3, 4]:
            result = wavelet_variance(x, level=level)
            assert len(result.period_bands) == level

    def test_total_is_float(self):
        x = RNG.standard_normal(32)
        result = wavelet_variance(x, level=2)
        assert isinstance(result.total, float)

    def test_band_localization_period4(self):
        """Pure period-4 signal concentrates power in the [2,4] band (j=1)."""
        T = 128
        t = np.arange(T)
        x = np.cos(2 * np.pi / 4.0 * t)  # period = 4
        result = wavelet_variance(x, level=5)
        # Dominant band should be j=1 → (2,4)
        dominant_idx = int(np.argmax(result.variance_per_scale))
        assert result.period_bands[dominant_idx] == (2, 4), (
            f"Period-4 signal: dominant band={result.period_bands[dominant_idx]}, expected (2,4)"
        )

    def test_band_localization_period8(self):
        """Pure period-8 signal concentrates power in the [4,8] band (j=2)."""
        T = 128
        t = np.arange(T)
        x = np.cos(2 * np.pi / 8.0 * t)  # period = 8
        result = wavelet_variance(x, level=5)
        dominant_idx = int(np.argmax(result.variance_per_scale))
        assert result.period_bands[dominant_idx] == (4, 8), (
            f"Period-8 signal: dominant band={result.period_bands[dominant_idx]}, expected (4,8)"
        )

    def test_result_frozen(self):
        """WaveletVarianceResult is a frozen dataclass."""
        x = RNG.standard_normal(32)
        result = wavelet_variance(x, level=2)
        with pytest.raises((AttributeError, TypeError)):
            result.variance_per_scale = np.zeros(2)  # type: ignore[misc]

    def test_summary_returns_string(self):
        x = RNG.standard_normal(64)
        result = wavelet_variance(x, level=3)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_total(self):
        """Summary string should mention total variance."""
        x = RNG.standard_normal(64)
        result = wavelet_variance(x, level=3)
        s = result.summary()
        assert "total" in s.lower() or str(round(result.total, 3)) in s

    def test_list_input_accepted(self):
        """Python list input is coerced to ndarray."""
        x = list(RNG.standard_normal(32))
        result = wavelet_variance(x, level=3)
        assert isinstance(result, WaveletVarianceResult)


# ===========================================================================
# 5. wavelet_coherence — properties and DGP tests
# ===========================================================================

class TestWaveletCoherence:
    def test_output_type_is_result(self):
        x = RNG.standard_normal(64)
        y = RNG.standard_normal(64)
        result = wavelet_coherence(x, y, level=3)
        assert isinstance(result, WaveletCoherenceResult)

    def test_coherence_shape(self):
        """coherence_per_scale has shape (level,)."""
        x = RNG.standard_normal(64)
        y = RNG.standard_normal(64)
        for level in [2, 3, 4]:
            result = wavelet_coherence(x, y, level=level)
            assert result.coherence_per_scale.shape == (level,)

    def test_coherence_in_minus1_plus1(self):
        """Coherence values must lie in [-1, 1]."""
        rng = np.random.default_rng(300)
        x = rng.standard_normal(128)
        y = rng.standard_normal(128)
        result = wavelet_coherence(x, y, level=4)
        assert (result.coherence_per_scale >= -1.0 - 1e-12).all()
        assert (result.coherence_per_scale <= 1.0 + 1e-12).all()

    def test_self_coherence_is_one(self):
        """Coherence of a signal with itself must be +1 at every scale."""
        rng = np.random.default_rng(301)
        x = rng.standard_normal(64)
        result = wavelet_coherence(x, x, level=3)
        np.testing.assert_allclose(
            result.coherence_per_scale, 1.0, atol=1e-12,
            err_msg="Self-coherence must be 1.0 at every scale"
        )

    def test_negated_coherence_is_minus_one(self):
        """Coherence of x with -x must be -1 at every scale."""
        rng = np.random.default_rng(302)
        x = rng.standard_normal(64)
        result = wavelet_coherence(x, -x, level=3)
        np.testing.assert_allclose(
            result.coherence_per_scale, -1.0, atol=1e-12,
            err_msg="Coherence(x, -x) must be -1.0 at every scale"
        )

    def test_near_identical_signals_coherence_near_one(self):
        """x + tiny noise: coherence should be close to 1 at all scales."""
        rng = np.random.default_rng(303)
        x = rng.standard_normal(128)
        y = x + 0.01 * rng.standard_normal(128)
        result = wavelet_coherence(x, y, level=3)
        assert (result.coherence_per_scale > 0.99).all(), (
            f"Near-identical signals: coherence={result.coherence_per_scale}, expected > 0.99"
        )

    def test_different_length_raises_value_error(self):
        """Mismatched lengths must raise ValueError."""
        x = np.ones(10)
        y = np.ones(11)
        with pytest.raises(ValueError, match="same length"):
            wavelet_coherence(x, y, level=2)

    def test_period_bands_match_variance_period_bands(self):
        """period_bands should match those of wavelet_variance for same level."""
        rng = np.random.default_rng(304)
        x = rng.standard_normal(64)
        y = rng.standard_normal(64)
        coh = wavelet_coherence(x, y, level=4)
        var = wavelet_variance(x, level=4)
        assert coh.period_bands == var.period_bands

    def test_n_bands_matches_level(self):
        x = RNG.standard_normal(64)
        y = RNG.standard_normal(64)
        for level in [2, 3, 4]:
            result = wavelet_coherence(x, y, level=level)
            assert len(result.period_bands) == level

    def test_result_frozen(self):
        """WaveletCoherenceResult is a frozen dataclass."""
        x = RNG.standard_normal(32)
        y = RNG.standard_normal(32)
        result = wavelet_coherence(x, y, level=2)
        with pytest.raises((AttributeError, TypeError)):
            result.coherence_per_scale = np.zeros(2)  # type: ignore[misc]

    def test_summary_returns_string(self):
        x = RNG.standard_normal(64)
        y = RNG.standard_normal(64)
        result = wavelet_coherence(x, y, level=3)
        s = result.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_coherence_keyword(self):
        x = RNG.standard_normal(64)
        y = RNG.standard_normal(64)
        result = wavelet_coherence(x, y, level=3)
        s = result.summary()
        assert "coherence" in s.lower()

    def test_symmetry_in_absolute_value(self):
        """Coherence is a correlation: |corr(x,y)| == |corr(y,x)| at each scale."""
        rng = np.random.default_rng(305)
        x = rng.standard_normal(64)
        y = rng.standard_normal(64)
        coh_xy = wavelet_coherence(x, y, level=3)
        coh_yx = wavelet_coherence(y, x, level=3)
        np.testing.assert_allclose(
            np.abs(coh_xy.coherence_per_scale),
            np.abs(coh_yx.coherence_per_scale),
            atol=1e-13,
        )

    def test_list_inputs_accepted(self):
        x = list(RNG.standard_normal(32))
        y = list(RNG.standard_normal(32))
        result = wavelet_coherence(x, y, level=2)
        assert isinstance(result, WaveletCoherenceResult)


# ===========================================================================
# 6. WaveletVarianceResult and WaveletCoherenceResult dataclasses
# ===========================================================================

class TestDataclasses:
    def test_wavelet_variance_result_fields_accessible(self):
        vs = np.array([0.1, 0.2, 0.3])
        bands = [(2, 4), (4, 8), (8, 16)]
        r = WaveletVarianceResult(
            variance_per_scale=vs,
            period_bands=bands,
            total=0.6,
            share=vs / 0.6,
        )
        np.testing.assert_array_equal(r.variance_per_scale, vs)
        assert r.period_bands == bands
        assert r.total == 0.6
        assert isinstance(r.share, np.ndarray)

    def test_wavelet_coherence_result_fields_accessible(self):
        coh = np.array([0.5, 0.8, -0.3])
        bands = [(2, 4), (4, 8), (8, 16)]
        r = WaveletCoherenceResult(coherence_per_scale=coh, period_bands=bands)
        np.testing.assert_array_equal(r.coherence_per_scale, coh)
        assert r.period_bands == bands

    def test_wavelet_variance_result_frozen(self):
        r = WaveletVarianceResult(
            variance_per_scale=np.array([0.1]),
            period_bands=[(2, 4)],
            total=0.1,
            share=np.array([1.0]),
        )
        with pytest.raises((AttributeError, TypeError)):
            r.total = 999.0  # type: ignore[misc]

    def test_wavelet_coherence_result_frozen(self):
        r = WaveletCoherenceResult(
            coherence_per_scale=np.array([0.5]),
            period_bands=[(2, 4)],
        )
        with pytest.raises((AttributeError, TypeError)):
            r.period_bands = [(99, 100)]  # type: ignore[misc]


# ===========================================================================
# 7. Integration / slow tests
# ===========================================================================

@pytest.mark.slow
def test_wavelet_variance_sum_of_shares_close_to_one_many_levels():
    """With many levels on a long series, the detail coefficients should
    capture most of the total variance (share sum approaches 1)."""
    rng = np.random.default_rng(500)
    T = 512
    x = rng.standard_normal(T)
    result = wavelet_variance(x, level=8)
    # With 8 levels and T=512 we capture most of the spectrum
    assert result.share.sum() > 0.85, (
        f"Expected share sum > 0.85, got {result.share.sum():.4f}"
    )


@pytest.mark.slow
def test_coherent_AR1_pair_has_high_coherence_at_long_scales():
    """Two AR(1) series driven by 80% common shock should have higher coherence
    at longer scales than independent noise."""
    rng = np.random.default_rng(501)
    T = 256
    # Common shock with persistence
    shock = np.zeros(T)
    for t in range(1, T):
        shock[t] = 0.8 * shock[t - 1] + rng.standard_normal()
    x = shock + 0.2 * rng.standard_normal(T)
    y = shock + 0.2 * rng.standard_normal(T)
    result = wavelet_coherence(x, y, level=5)
    # Long-scale (j=5, period [16,32]) coherence should be high
    long_scale_coh = result.coherence_per_scale[-1]  # coarsest scale
    fine_scale_noise_coh = result.coherence_per_scale[0]  # finest scale
    # Longest scale coherence should be noticeably higher than finest-scale
    assert long_scale_coh > fine_scale_noise_coh, (
        f"Long-scale coherence ({long_scale_coh:.3f}) should exceed "
        f"fine-scale coherence ({fine_scale_noise_coh:.3f})"
    )
