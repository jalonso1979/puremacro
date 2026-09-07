"""Pairwise multiresolution wavelet coherence.

Computes scale-by-scale correlations of MODWT detail coefficients,
with optional boundary-safe coefficient truncation.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

__all__ = ["WaveletCoherenceResult", "wavelet_coherence"]


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


def wavelet_coherence(
    x: np.ndarray,
    y: np.ndarray,
    level: int = 5,
    boundary_safe: bool = True,
) -> WaveletCoherenceResult:
    """Pairwise wavelet coherence per scale.

    Computes scale-by-scale correlations of the MODWT detail
    coefficients of x and y.

    Parameters
    ----------
    x : (T,) ndarray
        First input series.
    y : (T,) ndarray
        Second input series (same length as x).
    level : int, default 5
        Decomposition depth.
    boundary_safe : bool, default True
        If True, discards the 2^j - 1 boundary-affected coefficients
        at scale j to prevent circular convolution edge artifacts.

    Returns
    -------
    WaveletCoherenceResult
        Result containing coherence_per_scale and period_bands.
    """
    from puremacro.wavelet import modwt_haar

    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("x and y must have the same length.")
    cx = modwt_haar(x - x.mean(), level=level)
    cy = modwt_haar(y - y.mean(), level=level)
    coh = []
    bands = []
    for j, dx, dy in zip(range(level, 0, -1), cx[1:], cy[1:]):
        if boundary_safe:
            nb = 2 ** j - 1
            n_valid = len(dx) - nb
            if n_valid <= 0:
                coh.append(0.0)
                bands.append((2 ** j, 2 ** (j + 1)))
                continue
            keep = slice(nb, None)
        else:
            if len(dx) <= 0:
                coh.append(0.0)
                bands.append((2 ** j, 2 ** (j + 1)))
                continue
            keep = slice(None)
        a = dx[keep] - dx[keep].mean()
        b = dy[keep] - dy[keep].mean()
        sx = float((a ** 2).mean())
        sy = float((b ** 2).mean())
        sxy = float((a * b).mean())
        coh.append(sxy / np.sqrt(sx * sy) if sx > 0 and sy > 0 else 0.0)
        bands.append((2 ** j, 2 ** (j + 1)))
    return WaveletCoherenceResult(
        coherence_per_scale=np.array(coh[::-1]),
        period_bands=bands[::-1],
    )
