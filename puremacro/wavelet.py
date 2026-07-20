"""Wavelet decomposition for time-frequency analysis.

Implements the discrete wavelet transform (DWT) and the maximal-
overlap DWT (MODWT) with the Haar wavelet — pure-numpy, no PyWavelets.
The MODWT preserves time alignment across scales (no downsampling),
which is what business-cycle decompositions need: the level-j detail
coefficients capture power in the period band [2^j, 2^{j+1}].

The wavelet variance decomposition partitions Var(x) across scales,
giving a "where is the variance?" picture analogous to a spectral
density but localised in time.

References
----------
Percival, D.B. and Walden, A.T. (2000). Wavelet Methods for Time
    Series Analysis. Cambridge University Press.
Aguiar-Conraria, L. and Soares, M.J. (2014). The continuous wavelet
    transform: moving beyond uni- and bivariate analysis. JES 28(2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class WaveletVarianceResult:
    """Result of :func:`wavelet_variance`.

    Attributes
    ----------
    variance_per_scale : ndarray, shape (level,)
        Variance attributable to scale ``j`` (``j=1`` is the shortest
        period; ``j=level`` is the longest).
    period_bands : list[tuple[int, int]]
        Period ranges in observations, ordered j=1 first.
    total : float
        Total signal variance after demeaning.
    share : ndarray, shape (level,)
        Fraction of total variance per scale.
    """

    variance_per_scale: np.ndarray
    period_bands: list
    total: float
    share: np.ndarray

    def summary(self) -> str:
        return (
            f"Wavelet variance decomposition\n"
            f"  scales        : {len(self.variance_per_scale)}\n"
            f"  total variance: {self.total:.4f}\n"
            f"  top band      : period {self.period_bands[int(np.argmax(self.share))]}, "
            f"share {float(self.share.max()):.2%}\n"
        )


@dataclass(frozen=True)
class WaveletCoherenceResult:
    """Result of :func:`wavelet_coherence`.

    Attributes
    ----------
    coherence_per_scale : ndarray, shape (level,)
        Per-scale correlation of the MODWT detail coefficients of *x*
        and *y* (j=1 first).
    period_bands : list[tuple[int, int]]
        Period ranges in observations, ordered j=1 first.
    """

    coherence_per_scale: np.ndarray
    period_bands: list

    def summary(self) -> str:
        return (
            f"Wavelet coherence\n"
            f"  scales        : {len(self.coherence_per_scale)}\n"
            f"  max coherence : {float(self.coherence_per_scale.max()):+.3f} "
            f"at band {self.period_bands[int(np.argmax(self.coherence_per_scale))]}\n"
        )


# Haar wavelet filters.
_HAAR_LOW = np.array([1.0, 1.0]) / np.sqrt(2.0)
_HAAR_HIGH = np.array([1.0, -1.0]) / np.sqrt(2.0)


def _convolve_circular(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Circular convolution of x with h, length-preserving."""
    n = len(x)
    L = len(h)
    # Implement as FFT for speed (even though small).
    x_pad = x
    if L > n:
        # Pad x to length L via wraparound.
        x_pad = np.concatenate([x] * (int(np.ceil(L / n)) + 1))[:max(n, L)]
        n_eff = len(x_pad)
    else:
        n_eff = n
    Y = np.fft.fft(np.concatenate([h, np.zeros(n_eff - L)]))
    X = np.fft.fft(np.concatenate([x_pad, np.zeros(n_eff - len(x_pad))]))
    out = np.real(np.fft.ifft(X * Y))
    return out[:n]


def dwt_haar(x: np.ndarray, level: int | None = None) -> list[np.ndarray]:
    """Multi-level (decimated) Haar DWT.

    Parameters
    ----------
    x : (T,) ndarray. Length will be truncated to the nearest power of 2
        from above so each step halves cleanly.
    level : decomposition depth. Defaults to floor(log2(T)).

    Returns
    -------
    [cA_J, cD_J, cD_{J-1}, ..., cD_1] — approximation at the deepest
    level followed by details from coarsest to finest.
    """
    x = np.asarray(x, dtype=float).ravel()
    if level is None:
        level = max(1, int(np.floor(np.log2(len(x)))) - 1)
    coeffs = []
    cA = x.copy()
    for _ in range(level):
        n_pair = (len(cA) // 2) * 2
        cA = cA[:n_pair]
        cA_new = (cA[::2] + cA[1::2]) / np.sqrt(2.0)
        cD = (cA[::2] - cA[1::2]) / np.sqrt(2.0)
        coeffs.append(cD)
        cA = cA_new
    coeffs.append(cA)
    return list(reversed(coeffs))


def idwt_haar(coeffs: Sequence[np.ndarray]) -> np.ndarray:
    """Inverse multi-level Haar DWT."""
    cA = coeffs[0]
    for cD in coeffs[1:]:
        n = len(cA)
        out = np.empty(2 * n)
        out[::2] = (cA + cD) / np.sqrt(2.0)
        out[1::2] = (cA - cD) / np.sqrt(2.0)
        cA = out
    return cA


def modwt_haar(x: np.ndarray, level: int) -> list[np.ndarray]:
    """Maximal-overlap Haar DWT (à trous). Each cD_j has length T —
    time-aligned with x — so detail coefficients can be plotted
    against the calendar axis directly.

    Returns
    -------
    [cA_J, cD_J, cD_{J-1}, ..., cD_1] — same ordering as ``dwt_haar``.
    """
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    coeffs = []
    cA = x.copy()
    for j in range(1, level + 1):
        # MODWT filter coefficients at level j: insert 2^{j-1}-1 zeros
        # between the original Haar taps and rescale by 1/2^{j/2}.
        gap = 2 ** (j - 1)
        h = np.zeros((len(_HAAR_HIGH) - 1) * gap + 1)
        h[::gap] = _HAAR_HIGH / 2 ** (1 / 2)
        g = np.zeros((len(_HAAR_LOW) - 1) * gap + 1)
        g[::gap] = _HAAR_LOW / 2 ** (1 / 2)
        cD = _convolve_circular(cA, h)
        cA = _convolve_circular(cA, g)
        coeffs.append(cD)
    coeffs.append(cA)
    return list(reversed(coeffs))


def wavelet_variance(
    x: np.ndarray,
    level: int,
) -> WaveletVarianceResult:
    """Per-scale variance decomposition via the MODWT.

    Returns
    -------
    WaveletVarianceResult with ``variance_per_scale``, ``period_bands``,
    ``total``, and ``share``.
    """
    x = np.asarray(x, dtype=float).ravel()
    x_dem = x - x.mean()
    coeffs = modwt_haar(x_dem, level=level)  # [cA_J, cD_J, ..., cD_1]
    # The detail coefficients at each scale j contribute variance
    # E[cD_j^2] up to a normalisation. For the MODWT-Haar with the
    # 2^{-j/2} rescaling above, the wavelet variance estimator is the
    # mean of the squared detail coefficients.
    var_per_scale_list: list[float] = []
    bands: list[tuple[int, int]] = []
    for j, cD in zip(range(level, 0, -1), coeffs[1:]):
        var_per_scale_list.append(float((cD ** 2).mean()))
        bands.append((2 ** j, 2 ** (j + 1)))
    var_per_scale: np.ndarray = np.array(var_per_scale_list[::-1])  # j=1 first
    bands = bands[::-1]
    total = float((x_dem ** 2).mean())
    share: np.ndarray = var_per_scale / total if total > 0 else np.zeros_like(var_per_scale)
    return WaveletVarianceResult(
        variance_per_scale=var_per_scale,
        period_bands=bands,
        total=total,
        share=share,
    )


def wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    level: int,
) -> WaveletCoherenceResult:
    """Pairwise wavelet coherence per scale.

    Computes scale-by-scale correlations of the MODWT detail
    coefficients of x and y. Useful for asking "at the 2-4 year
    business-cycle scale, do x and y co-move?"

    Returns
    -------
    WaveletCoherenceResult with ``coherence_per_scale`` and ``period_bands``.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    cx = modwt_haar(x - x.mean(), level=level)
    cy = modwt_haar(y - y.mean(), level=level)
    coh = []
    bands = []
    for j, dx, dy in zip(range(level, 0, -1), cx[1:], cy[1:]):
        sx = float((dx ** 2).mean()); sy = float((dy ** 2).mean())
        sxy = float((dx * dy).mean())
        coh.append(sxy / np.sqrt(sx * sy) if sx > 0 and sy > 0 else 0.0)
        bands.append((2 ** j, 2 ** (j + 1)))
    return WaveletCoherenceResult(
        coherence_per_scale=np.array(coh[::-1]),
        period_bands=bands[::-1],
    )


__all__ = [
    "dwt_haar", "idwt_haar", "modwt_haar",
    "wavelet_variance", "wavelet_coherence",
    "WaveletVarianceResult", "WaveletCoherenceResult",
]
