"""Frequency-domain tools: Welch periodogram, cross-spectrum,
coherence, gain, phase.

Self-contained — uses only numpy.fft, so this remains in the
Pyodide-friendly subset (no scipy.signal dependency required for the
core estimators).

Reference
---------
Welch, P.D. (1967). The use of fast Fourier transform for the
    estimation of power spectra: a method based on time averaging
    over short, modified periodograms. IEEE Trans. AU-15, 70-73.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WelchCrossResult:
    """Result of :func:`welch_cross`.

    Attributes
    ----------
    f : ndarray
        Frequency grid (Hz).
    Pxx, Pyy : ndarray
        Auto-spectra.
    Pxy : ndarray (complex)
        Cross-spectrum.
    coherence : ndarray
        ``|Pxy|^2 / (Pxx Pyy)``.
    gain : ndarray
        ``|Pxy| / Pxx`` — magnitude of the frequency response x → y.
    phase : ndarray
        ``angle(Pxy)`` in radians.
    """

    f: np.ndarray
    Pxx: np.ndarray
    Pyy: np.ndarray
    Pxy: np.ndarray
    coherence: np.ndarray
    gain: np.ndarray
    phase: np.ndarray

    def summary(self) -> str:
        peak = int(np.argmax(self.coherence))
        return (
            f"Welch cross-spectrum\n"
            f"  n_freq          : {len(self.f)}\n"
            f"  peak coherence  : {float(self.coherence[peak]):.3f} at f={self.f[peak]:.4f}\n"
        )


def _hann(n: int) -> np.ndarray:
    """Hann window of length n."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / max(n - 1, 1))


def _welch_segments(x: np.ndarray, n_seg: int, overlap: float):
    """Iterate (start, stop) for half-overlap Welch segments."""
    T = len(x)
    step = max(1, int(round(n_seg * (1 - overlap))))
    starts = list(range(0, T - n_seg + 1, step))
    return starts


def welch_psd(
    x: np.ndarray,
    fs: float = 1.0,
    n_seg: int = 64,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Welch power spectral density.

    Parameters
    ----------
    x : (T,) ndarray.
    fs : sampling frequency (in Hz; use 4 for quarterly cycles/year).
    n_seg : segment length.
    overlap : fraction in [0, 1).

    Returns
    -------
    f : (n_seg//2 + 1,) frequency grid in Hz.
    Pxx : (n_seg//2 + 1,) PSD estimate.
    """
    x = np.asarray(x, dtype=float).ravel()
    win = _hann(n_seg)
    starts = _welch_segments(x, n_seg, overlap)
    if not starts:
        raise ValueError("Series too short for the requested n_seg.")
    norm = (win ** 2).sum() * fs
    psd_acc = np.zeros(n_seg // 2 + 1)
    for s in starts:
        seg = (x[s:s + n_seg] - x[s:s + n_seg].mean()) * win
        spec = np.fft.rfft(seg)
        psd_acc += (np.abs(spec) ** 2) / norm
    psd_acc /= len(starts)
    f = np.fft.rfftfreq(n_seg, d=1.0 / fs)
    return f, psd_acc


def welch_cross(
    x: np.ndarray,
    y: np.ndarray,
    fs: float = 1.0,
    n_seg: int = 64,
    overlap: float = 0.5,
) -> WelchCrossResult:
    """Welch estimates of auto- and cross-spectra plus coherence,
    gain, and phase.

    Returns
    -------
    WelchCrossResult with ``f``, ``Pxx``, ``Pyy``, ``Pxy``, ``coherence``,
    ``gain``, and ``phase``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    win = _hann(n_seg)
    starts = _welch_segments(x, n_seg, overlap)
    norm = (win ** 2).sum() * fs
    n_freq = n_seg // 2 + 1
    Pxx = np.zeros(n_freq)
    Pyy = np.zeros(n_freq)
    Pxy = np.zeros(n_freq, dtype=complex)
    for s in starts:
        sx = (x[s:s + n_seg] - x[s:s + n_seg].mean()) * win
        sy = (y[s:s + n_seg] - y[s:s + n_seg].mean()) * win
        Sx = np.fft.rfft(sx)
        Sy = np.fft.rfft(sy)
        Pxx += (np.abs(Sx) ** 2) / norm
        Pyy += (np.abs(Sy) ** 2) / norm
        Pxy += (np.conj(Sx) * Sy) / norm
    Pxx /= len(starts); Pyy /= len(starts); Pxy /= len(starts)
    f = np.fft.rfftfreq(n_seg, d=1.0 / fs)
    coherence = (np.abs(Pxy) ** 2) / np.maximum(Pxx * Pyy, 1e-20)
    gain = np.abs(Pxy) / np.maximum(Pxx, 1e-20)
    phase = np.angle(Pxy)
    return WelchCrossResult(
        f=f,
        Pxx=Pxx, Pyy=Pyy, Pxy=Pxy,
        coherence=coherence,
        gain=gain,
        phase=phase,
    )


def business_cycle_band_power(
    x: np.ndarray,
    fs: float = 1.0,
    low_period: float = 6.0,
    high_period: float = 32.0,
    n_seg: int = 64,
) -> float:
    """Fraction of variance of x in the business-cycle band [low_period,
    high_period].

    Convention: ``low_period`` and ``high_period`` are expressed in the
    *same time units as 1/fs*. The default ``fs=1`` treats one
    observation as one period — so the default 6–32 corresponds to the
    Burns-Mitchell business-cycle band of 6–32 quarters when applied to
    quarterly data, or 6–32 months for monthly data. If you prefer
    "cycles per year" with quarterly data, pass ``fs=4`` and supply
    ``low_period`` / ``high_period`` in years (e.g. 1.5 and 8).
    """
    f, Pxx = welch_psd(x, fs=fs, n_seg=n_seg)
    df = f[1] - f[0] if len(f) > 1 else 1.0
    f_low = 1.0 / high_period
    f_high = 1.0 / low_period
    band = (f >= f_low) & (f <= f_high)
    total = float(np.sum(Pxx) * df)
    band_power = float(np.sum(Pxx[band]) * df)
    return band_power / total if total > 0 else 0.0


__all__ = [
    "welch_psd", "welch_cross", "business_cycle_band_power",
    "WelchCrossResult",
]
