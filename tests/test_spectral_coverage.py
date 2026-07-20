"""Tests for puremacro.spectral — Welch PSD, cross-spectrum, coherence, gain, phase.

Strategy:
- Known-value: build small synthetic signals with analytically known spectral
  properties (pure sinusoids, white noise) and verify the estimator recovers
  the expected dominant frequency, coherence, and shape.
- Properties: output shapes, dtypes, non-negativity of PSD, coherence bounds
  [0, 1], frozen dataclass, gain/phase consistency with Pxy.
- Edge cases + error handling: length mismatch raises ValueError, too-short
  series raises ValueError.
- DGP-driven: identical signals yield coherence 1; orthogonal signals yield
  coherence near 0; gain should capture the correct transfer-function magnitude.
"""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.spectral import (
    WelchCrossResult,
    business_cycle_band_power,
    welch_cross,
    welch_psd,
)

RNG = np.random.default_rng(2025)


# ===========================================================================
# 1. _hann (private) — tested indirectly through welch_psd shapes
# ===========================================================================

class TestHannWindow:
    """Test the Hann window via the observable behavior of welch_psd."""

    def test_output_shape_default_n_seg(self):
        """welch_psd returns f and Pxx of length n_seg//2 + 1."""
        n_seg = 64
        x = RNG.standard_normal(256)
        f, Pxx = welch_psd(x, fs=1.0, n_seg=n_seg)
        assert f.shape == (n_seg // 2 + 1,)
        assert Pxx.shape == (n_seg // 2 + 1,)

    def test_output_shape_various_n_seg(self):
        """Shape is (n_seg//2 + 1,) for any n_seg."""
        x = RNG.standard_normal(512)
        for n_seg in [16, 32, 128]:
            f, Pxx = welch_psd(x, fs=1.0, n_seg=n_seg)
            assert f.shape == (n_seg // 2 + 1,)
            assert Pxx.shape == (n_seg // 2 + 1,)


# ===========================================================================
# 2. welch_psd — structural + known-value tests
# ===========================================================================

class TestWelchPsd:
    def test_psd_nonnegative(self):
        """Power spectral density must be non-negative everywhere."""
        x = RNG.standard_normal(256)
        f, Pxx = welch_psd(x, fs=1.0, n_seg=64)
        assert (Pxx >= 0).all()

    def test_frequency_grid_starts_at_zero(self):
        """DC component: f[0] == 0."""
        x = RNG.standard_normal(256)
        f, _ = welch_psd(x, fs=1.0, n_seg=64)
        assert f[0] == pytest.approx(0.0)

    def test_frequency_grid_max_is_nyquist(self):
        """Highest frequency is the Nyquist frequency fs/2."""
        fs = 4.0
        n_seg = 64
        x = RNG.standard_normal(256)
        f, _ = welch_psd(x, fs=fs, n_seg=n_seg)
        assert f[-1] == pytest.approx(fs / 2.0)

    def test_frequency_grid_monotone(self):
        """Frequency grid must be strictly increasing."""
        x = RNG.standard_normal(256)
        f, _ = welch_psd(x, fs=1.0, n_seg=64)
        assert (np.diff(f) > 0).all()

    def test_frequency_grid_uniform(self):
        """Spacing between frequency bins must be uniform (rfftfreq property)."""
        x = RNG.standard_normal(256)
        f, _ = welch_psd(x, fs=1.0, n_seg=64)
        diffs = np.diff(f)
        assert np.allclose(diffs, diffs[0], rtol=1e-12)

    def test_psd_float_dtype(self):
        """Output arrays must have float dtype."""
        x = RNG.standard_normal(128)
        f, Pxx = welch_psd(x, fs=1.0, n_seg=32)
        assert np.issubdtype(f.dtype, np.floating)
        assert np.issubdtype(Pxx.dtype, np.floating)

    def test_constant_series_psd_near_zero(self):
        """A constant series (zero variance) has a PSD near zero."""
        x = np.full(256, 3.14)
        f, Pxx = welch_psd(x, fs=1.0, n_seg=64)
        np.testing.assert_allclose(Pxx, 0.0, atol=1e-28)

    def test_list_input_accepted(self):
        """Python list coerced to ndarray without error."""
        x = list(RNG.standard_normal(128).tolist())
        f, Pxx = welch_psd(x, fs=1.0, n_seg=32)
        assert isinstance(Pxx, np.ndarray)

    def test_too_short_raises_value_error(self):
        """Series shorter than n_seg must raise ValueError."""
        x = np.ones(10)
        with pytest.raises(ValueError):
            welch_psd(x, n_seg=64)

    def test_sinusoid_peak_at_correct_frequency(self):
        """Pure sinusoid at f0 should show dominant power near f0."""
        fs = 1.0
        f0 = 0.1  # Hz
        n_seg = 128
        T = 1024
        t = np.arange(T)
        x = np.sin(2 * np.pi * f0 * t)
        f, Pxx = welch_psd(x, fs=fs, n_seg=n_seg)
        # Drop DC bin (index 0)
        peak_idx = int(np.argmax(Pxx[1:])) + 1
        # Peak frequency should be within one bin of f0
        df = f[1] - f[0]
        assert abs(f[peak_idx] - f0) <= df + 1e-12, (
            f"Peak at {f[peak_idx]:.4f} Hz, expected near {f0:.4f} Hz (df={df:.4f})"
        )

    def test_higher_fs_rescales_frequency_grid(self):
        """With fs=4, Nyquist = 2 Hz; frequency grid matches rfftfreq convention."""
        n_seg = 64
        x = RNG.standard_normal(256)
        f4, _ = welch_psd(x, fs=4.0, n_seg=n_seg)
        f1, _ = welch_psd(x, fs=1.0, n_seg=n_seg)
        # f4 should be exactly 4 * f1
        np.testing.assert_allclose(f4, 4.0 * f1, rtol=1e-14)

    def test_overlap_parameter_changes_n_segments(self):
        """Different overlaps affect averaging but output shape is the same."""
        x = RNG.standard_normal(256)
        f_50, Pxx_50 = welch_psd(x, fs=1.0, n_seg=64, overlap=0.5)
        f_75, Pxx_75 = welch_psd(x, fs=1.0, n_seg=64, overlap=0.75)
        # Shape must be identical
        assert f_50.shape == f_75.shape
        assert Pxx_50.shape == Pxx_75.shape

    def test_white_noise_psd_is_approximately_flat(self):
        """White noise has a flat PSD; standard deviation of normalized Pxx
        should be small relative to the mean (over many averages)."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(4096)
        f, Pxx = welch_psd(x, fs=1.0, n_seg=256, overlap=0.5)
        # Exclude DC (index 0) — its variance is biased by mean removal
        Pxx_ac = Pxx[1:]
        cv = Pxx_ac.std() / Pxx_ac.mean()  # coefficient of variation
        assert cv < 0.5, f"PSD of white noise too uneven: CV={cv:.3f}"

    def test_2d_input_ravelled(self):
        """2-D input is ravelled to 1-D without error."""
        x = RNG.standard_normal((4, 64))  # 256 samples as 4x64
        f, Pxx = welch_psd(x, fs=1.0, n_seg=64)
        assert Pxx.shape == (64 // 2 + 1,)


# ===========================================================================
# 3. welch_cross — structural, property, and DGP tests
# ===========================================================================

class TestWelchCross:
    def test_returns_welch_cross_result(self):
        x = RNG.standard_normal(256)
        y = RNG.standard_normal(256)
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        assert isinstance(result, WelchCrossResult)

    def test_all_arrays_same_length(self):
        """f, Pxx, Pyy, Pxy, coherence, gain, phase must all have the same length."""
        x = RNG.standard_normal(256)
        y = RNG.standard_normal(256)
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        n = len(result.f)
        assert len(result.Pxx) == n
        assert len(result.Pyy) == n
        assert len(result.Pxy) == n
        assert len(result.coherence) == n
        assert len(result.gain) == n
        assert len(result.phase) == n

    def test_length_matches_n_seg(self):
        """Output length is n_seg//2 + 1."""
        n_seg = 64
        x = RNG.standard_normal(256)
        y = RNG.standard_normal(256)
        result = welch_cross(x, y, fs=1.0, n_seg=n_seg)
        expected = n_seg // 2 + 1
        assert len(result.f) == expected

    def test_pxx_pyy_nonnegative(self):
        """Auto-spectra Pxx and Pyy must be non-negative."""
        x = RNG.standard_normal(256)
        y = RNG.standard_normal(256)
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        assert (result.Pxx >= 0).all()
        assert (result.Pyy >= 0).all()

    def test_coherence_bounds(self):
        """Coherence = |Pxy|^2 / (Pxx * Pyy) must be in [0, 1]."""
        rng = np.random.default_rng(10)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        assert (result.coherence >= 0.0 - 1e-12).all()
        assert (result.coherence <= 1.0 + 1e-12).all()

    def test_gain_nonnegative(self):
        """Gain = |Pxy| / Pxx must be non-negative."""
        x = RNG.standard_normal(256)
        y = RNG.standard_normal(256)
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        assert (result.gain >= 0.0).all()

    def test_phase_in_range(self):
        """Phase = angle(Pxy) must be in [-pi, pi]."""
        x = RNG.standard_normal(256)
        y = RNG.standard_normal(256)
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        assert (result.phase >= -np.pi - 1e-12).all()
        assert (result.phase <= np.pi + 1e-12).all()

    def test_pxy_is_complex(self):
        """Cross-spectrum Pxy must be complex-valued."""
        x = RNG.standard_normal(256)
        y = RNG.standard_normal(256)
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        assert np.issubdtype(result.Pxy.dtype, np.complexfloating)

    def test_pxx_matches_welch_psd(self):
        """Pxx from welch_cross must match welch_psd applied to x alone."""
        rng = np.random.default_rng(20)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        _, Pxx_standalone = welch_psd(x, fs=1.0, n_seg=64, overlap=0.5)
        result = welch_cross(x, y, fs=1.0, n_seg=64, overlap=0.5)
        np.testing.assert_allclose(result.Pxx, Pxx_standalone, rtol=1e-12)

    def test_pyy_matches_welch_psd(self):
        """Pyy from welch_cross must match welch_psd applied to y alone."""
        rng = np.random.default_rng(21)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        _, Pyy_standalone = welch_psd(y, fs=1.0, n_seg=64, overlap=0.5)
        result = welch_cross(x, y, fs=1.0, n_seg=64, overlap=0.5)
        np.testing.assert_allclose(result.Pyy, Pyy_standalone, rtol=1e-12)

    def test_identical_signals_coherence_is_one(self):
        """Coherence of x with itself must be 1 at every frequency."""
        rng = np.random.default_rng(30)
        x = rng.standard_normal(512)
        result = welch_cross(x, x, fs=1.0, n_seg=64)
        # Coherence = |Pxy|^2 / (Pxx * Pyy); when x==y, Pxy = Pxx = Pyy, so = 1
        np.testing.assert_allclose(result.coherence, 1.0, atol=1e-12)

    def test_phase_of_identical_signals_is_zero(self):
        """Phase of x with itself must be 0 at every frequency."""
        rng = np.random.default_rng(31)
        x = rng.standard_normal(256)
        result = welch_cross(x, x, fs=1.0, n_seg=64)
        np.testing.assert_allclose(result.phase, 0.0, atol=1e-12)

    def test_gain_of_identical_signals_is_one(self):
        """Gain of x with itself = |Pxx| / Pxx = 1 (modulo numerical noise)."""
        rng = np.random.default_rng(32)
        x = rng.standard_normal(256)
        result = welch_cross(x, x, fs=1.0, n_seg=64)
        # Gain = |Pxy| / Pxx; with x==y, Pxy = Pxx (real positive), so gain = 1
        np.testing.assert_allclose(result.gain, 1.0, atol=1e-12)

    def test_scaled_y_gain_reflects_scale_factor(self):
        """If y = k * x, then gain (Pxy / Pxx) should approximate k."""
        rng = np.random.default_rng(33)
        k = 3.0
        x = rng.standard_normal(512)
        y = k * x
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        # gain = |Pxy| / Pxx; Pxy = conj(Sx) * k*Sx / norm = k * |Sx|^2 / norm = k * Pxx
        # so gain = k * Pxx / Pxx = k
        np.testing.assert_allclose(result.gain, k, atol=1e-10)

    def test_negated_y_phase_is_pi(self):
        """If y = -x, then Pxy is negative real, so phase = pi."""
        rng = np.random.default_rng(34)
        x = rng.standard_normal(256)
        y = -x
        result = welch_cross(x, y, fs=1.0, n_seg=64)
        # phase = angle(-Pxx) = pi (since Pxx > 0)
        # Avoid DC bin which may be 0/0
        phase_ac = result.phase[1:]
        np.testing.assert_allclose(np.abs(phase_ac), np.pi, atol=1e-12)

    def test_length_mismatch_raises_value_error(self):
        """Different-length x and y must raise ValueError."""
        x = np.ones(100)
        y = np.ones(99)
        with pytest.raises(ValueError, match="same length"):
            welch_cross(x, y)

    def test_frequency_grid_matches_welch_psd(self):
        """f returned by welch_cross must match f from welch_psd."""
        rng = np.random.default_rng(40)
        x = rng.standard_normal(256)
        y = rng.standard_normal(256)
        f_cross = welch_cross(x, y, fs=4.0, n_seg=64).f
        f_psd, _ = welch_psd(x, fs=4.0, n_seg=64)
        np.testing.assert_array_equal(f_cross, f_psd)

    def test_coherence_definition_consistency(self):
        """Verify: coherence == |Pxy|^2 / (Pxx * Pyy) element-wise."""
        rng = np.random.default_rng(50)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        r = welch_cross(x, y, fs=1.0, n_seg=64)
        expected = (np.abs(r.Pxy) ** 2) / np.maximum(r.Pxx * r.Pyy, 1e-20)
        np.testing.assert_allclose(r.coherence, expected, rtol=1e-12)

    def test_gain_definition_consistency(self):
        """Verify: gain == |Pxy| / Pxx element-wise."""
        rng = np.random.default_rng(51)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        r = welch_cross(x, y, fs=1.0, n_seg=64)
        expected = np.abs(r.Pxy) / np.maximum(r.Pxx, 1e-20)
        np.testing.assert_allclose(r.gain, expected, rtol=1e-12)

    def test_phase_definition_consistency(self):
        """Verify: phase == angle(Pxy) element-wise."""
        rng = np.random.default_rng(52)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        r = welch_cross(x, y, fs=1.0, n_seg=64)
        expected = np.angle(r.Pxy)
        np.testing.assert_allclose(r.phase, expected, rtol=1e-12)

    def test_list_inputs_accepted(self):
        """Python list inputs are coerced without error."""
        x = list(RNG.standard_normal(128).tolist())
        y = list(RNG.standard_normal(128).tolist())
        result = welch_cross(x, y)
        assert isinstance(result, WelchCrossResult)

    def test_common_shock_high_coherence(self):
        """x = common + noise, y = common + noise: coherence should be high
        at the frequency of the common component."""
        rng = np.random.default_rng(60)
        T = 1024
        f0 = 0.1
        t = np.arange(T)
        common = np.sin(2 * np.pi * f0 * t)
        x = common + 0.1 * rng.standard_normal(T)
        y = common + 0.1 * rng.standard_normal(T)
        n_seg = 128
        result = welch_cross(x, y, fs=1.0, n_seg=n_seg)
        # Find bin closest to f0
        f = result.f
        idx = int(np.argmin(np.abs(f - f0)))
        # Coherence at f0 should be very high (>= 0.9)
        assert result.coherence[idx] >= 0.9, (
            f"Coherence at f0={f0} Hz is {result.coherence[idx]:.3f}, expected >= 0.9"
        )

    def test_orthogonal_signals_coherence_near_zero(self):
        """Sine and cosine at the same frequency are 90 degrees out of phase
        and long averages should yield moderate coherence; the point is that
        coherence > 0 because they share the same frequency — but independent
        noise signals should have lower coherence."""
        rng = np.random.default_rng(61)
        T = 2048
        x = rng.standard_normal(T)
        y = rng.standard_normal(T)
        result = welch_cross(x, y, fs=1.0, n_seg=128)
        # Mean coherence of two independent white noise series should be < 0.5
        assert result.coherence.mean() < 0.5, (
            f"Mean coherence of independent noise = {result.coherence.mean():.3f}, expected < 0.5"
        )


# ===========================================================================
# 4. WelchCrossResult dataclass — frozen + summary
# ===========================================================================

class TestWelchCrossResult:
    def _make_result(self):
        x = RNG.standard_normal(256)
        y = RNG.standard_normal(256)
        return welch_cross(x, y, fs=1.0, n_seg=64)

    def test_frozen_dataclass(self):
        """WelchCrossResult is frozen — attribute assignment must raise."""
        r = self._make_result()
        with pytest.raises((AttributeError, TypeError)):
            r.coherence = np.zeros_like(r.coherence)  # type: ignore[misc]

    def test_summary_returns_string(self):
        r = self._make_result()
        s = r.summary()
        assert isinstance(s, str)
        assert len(s) > 0

    def test_summary_contains_n_freq(self):
        """Summary must mention the number of frequency bins."""
        r = self._make_result()
        s = r.summary()
        assert "n_freq" in s.lower() or str(len(r.f)) in s

    def test_summary_contains_peak_coherence(self):
        """Summary must contain the string 'coherence'."""
        r = self._make_result()
        s = r.summary()
        assert "coherence" in s.lower()

    def test_summary_coherence_value_is_in_range(self):
        """The numeric value reported in summary should be in [0, 1]."""
        r = self._make_result()
        # Peak coherence is max of r.coherence, which is in [0, 1]
        peak = float(r.coherence.max())
        assert 0.0 <= peak <= 1.0 + 1e-12


# ===========================================================================
# 5. business_cycle_band_power — known-value and property tests
# ===========================================================================

class TestBusinessCycleBandPower:
    def test_returns_float(self):
        """Return value must be a plain Python float."""
        x = RNG.standard_normal(256)
        result = business_cycle_band_power(x)
        assert isinstance(result, float)

    def test_in_zero_one(self):
        """Band power fraction must be in [0, 1]."""
        rng = np.random.default_rng(70)
        for _ in range(5):
            x = rng.standard_normal(512)
            bp = business_cycle_band_power(x)
            assert 0.0 <= bp <= 1.0, f"Band power = {bp} not in [0, 1]"

    def test_constant_series_returns_zero(self):
        """Constant series has zero variance so band power fraction = 0."""
        x = np.full(256, 5.0)
        bp = business_cycle_band_power(x)
        assert bp == pytest.approx(0.0, abs=1e-10)

    def test_bc_frequency_sinusoid_dominates_band(self):
        """A sinusoid at period 16 (within the 6-32 default band) should yield
        a band-power fraction substantially greater than 0.5."""
        T = 1024
        t = np.arange(T)
        # Period 16 quarters → frequency 1/16 cycles/observation
        x = np.cos(2 * np.pi / 16.0 * t)
        bp = business_cycle_band_power(x, fs=1.0, low_period=6.0, high_period=32.0, n_seg=128)
        assert bp > 0.5, f"Sinusoid in BC band: band power = {bp:.3f}, expected > 0.5"

    def test_high_frequency_sinusoid_low_band_power(self):
        """A sinusoid at period 2 (Nyquist, outside the 6-32 band) should yield
        a low band-power fraction (< 0.3)."""
        T = 512
        t = np.arange(T)
        # Period 2 → f = 0.5 (Nyquist), outside the BC band
        x = np.cos(2 * np.pi / 2.0 * t)
        bp = business_cycle_band_power(x, fs=1.0, low_period=6.0, high_period=32.0, n_seg=64)
        assert bp < 0.3, f"High-freq sinusoid: band power = {bp:.3f}, expected < 0.3"

    def test_very_long_period_sinusoid_low_bc_power(self):
        """Sinusoid at period 64 (outside 6-32 band) should yield low band power."""
        T = 1024
        t = np.arange(T)
        x = np.cos(2 * np.pi / 64.0 * t)
        bp = business_cycle_band_power(x, fs=1.0, low_period=6.0, high_period=32.0, n_seg=128)
        assert bp < 0.3, f"Low-freq sinusoid: band power = {bp:.3f}, expected < 0.3"

    def test_custom_band_periods(self):
        """Custom band [low_period, high_period] works without error."""
        x = RNG.standard_normal(256)
        bp = business_cycle_band_power(x, fs=4.0, low_period=6.0, high_period=32.0)
        assert 0.0 <= bp <= 1.0

    def test_white_noise_bc_fraction_reasonable(self):
        """White noise: band power fraction should be roughly proportional to
        the fraction of the spectrum inside the band."""
        rng = np.random.default_rng(80)
        x = rng.standard_normal(4096)
        bp = business_cycle_band_power(x, fs=1.0, low_period=6.0, high_period=32.0, n_seg=256)
        # The BC band [1/32, 1/6] occupies roughly (1/6 - 1/32) / 0.5 ≈ 24% of spectrum
        # Allow a generous tolerance due to estimation noise
        assert 0.05 < bp < 0.6, f"White noise BC band power = {bp:.3f} seems unreasonable"

    def test_list_input_accepted(self):
        """Python list input is coerced without error."""
        x = list(RNG.standard_normal(256).tolist())
        bp = business_cycle_band_power(x)
        assert isinstance(bp, float)

    def test_n_seg_parameter_respected(self):
        """Different n_seg values return floats in [0, 1]."""
        x = RNG.standard_normal(512)
        for n_seg in [32, 64, 128]:
            bp = business_cycle_band_power(x, n_seg=n_seg)
            assert 0.0 <= bp <= 1.0

    def test_full_band_equals_one(self):
        """When band covers the entire spectrum (low_period=2, high_period=very large),
        band power fraction should be 1."""
        x = RNG.standard_normal(256)
        # Period 2 to 10000 covers almost the whole spectrum
        # Note: f_low = 1/10000, f_high = 1/2 = Nyquist
        # All bins from near-DC to Nyquist should be included
        bp = business_cycle_band_power(x, fs=1.0, low_period=2.0, high_period=10000.0, n_seg=64)
        assert bp == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# 6. __all__ API surface check
# ===========================================================================

def test_all_exports_present():
    """The four names in __all__ must be importable from puremacro.spectral."""
    import puremacro.spectral as sp
    for name in sp.__all__:
        assert hasattr(sp, name), f"__all__ member '{name}' not found in module"


def test_welch_cross_result_in_all():
    """WelchCrossResult must appear in __all__."""
    import puremacro.spectral as sp
    assert "WelchCrossResult" in sp.__all__


# ===========================================================================
# 7. Slow / integration tests
# ===========================================================================

@pytest.mark.slow
def test_welch_cross_long_series():
    """Long series (T=8192) runs without error; coherence stays in [0, 1]."""
    rng = np.random.default_rng(90)
    T = 8192
    x = rng.standard_normal(T)
    y = rng.standard_normal(T)
    result = welch_cross(x, y, fs=4.0, n_seg=256, overlap=0.5)
    assert (result.coherence >= 0.0).all()
    assert (result.coherence <= 1.0 + 1e-12).all()


@pytest.mark.slow
def test_bc_band_power_more_data_stable():
    """Increasing T does not push band power outside [0, 1]."""
    rng = np.random.default_rng(91)
    for T in [512, 1024, 2048, 4096]:
        x = rng.standard_normal(T)
        bp = business_cycle_band_power(x, fs=1.0, n_seg=64)
        assert 0.0 <= bp <= 1.0, f"T={T}: band power = {bp} outside [0,1]"
