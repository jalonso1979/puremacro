"""Σ-as-object: sector covariance matrix as the unit of analysis.

Aggregate output volatility under the BGP / sectoral-decomposition framing is

    Var(g) = w' · Σ · w,         Σ = diag(σ) · R · diag(σ)

where ``σ`` is a vector of per-sector volatilities and ``R`` is a correlation
matrix estimated from rolling windows of sector growth rates. Treating Σ as
an object lets the caller swap weights ``w`` cheaply, decompose into
diagonal vs. off-diagonal contributions, summarise comovement structure,
and produce risk-accounting tables — all without re-estimating Σ.

The "covariance premium"

    cov_share(w) = (w'Σw − Σ_s w_s² σ_s²) / (w'Σw)

isolates the share of aggregate variance that comes from cross-sector
comovement rather than sector-specific volatility. This is the headline
metric for the sectoral / labor-volatility track in the parent MAV
workspace; see ``sigma_object_extended_ulysses.md`` for the narrative
motivation.

This module is the Python port of ``MAV/SigmaObject.m``. It preserves the
MATLAB API (``var``, ``std``, ``var_no_cov``, ``cov_premium_var``,
``cov_share``, ``diag_contrib``, ``pair_contrib``, ``mean_corr``,
``pc1_share``, ``summary``) so master scripts can be translated
line-for-line.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd


def project_psd(M: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone by clipping eigenvalues.

    Parameters
    ----------
    M : (n, n) ndarray
        Symmetric (or near-symmetric) matrix to project.
    tol : float, default 1e-10
        Eigenvalues below this are clipped to zero. ``1e-10`` matches the
        MATLAB default; numerically-zero negative eigenvalues from
        round-off are removed without touching genuine signal.

    Returns
    -------
    ndarray
        ``(n, n)`` symmetric PSD matrix.
    """
    M = (M + M.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = eigvals.real
    eigvals = np.where(eigvals < tol, 0.0, eigvals)
    out = eigvecs @ np.diag(eigvals) @ eigvecs.T
    out = (out + out.T) / 2.0
    return out


def clean_correlation(
    R: np.ndarray,
    *,
    enforce_psd: bool = True,
    tol: float = 1e-10,
) -> np.ndarray:
    """Make ``R`` look like a proper correlation matrix.

    Steps:
      1. Force symmetry.
      2. Replace NaNs with 0 off-diagonal, 1 on diagonal.
      3. Force diag = 1.
      4. (optional) PSD projection + renormalisation back to corr.

    Parameters
    ----------
    R : (n, n) ndarray
        Candidate correlation matrix.
    enforce_psd : bool, default True
        If True, project onto the PSD cone and renormalise so the
        diagonal is exactly 1.
    tol : float, default 1e-10
        Eigenvalue clip threshold passed to :func:`project_psd`.

    Returns
    -------
    ndarray
        ``(n, n)`` symmetric matrix with unit diagonal (and PSD if
        ``enforce_psd``).
    """
    R = np.asarray(R, dtype=float)
    R = (R + R.T) / 2.0
    if np.isnan(R).any():
        R = np.where(np.isnan(R), 0.0, R)
    n = R.shape[0]
    R[np.diag_indices(n)] = 1.0

    if enforce_psd:
        R = project_psd(R, tol)
        d = np.sqrt(np.maximum(np.diag(R), tol))
        R = R / np.outer(d, d)
        R[np.diag_indices(n)] = 1.0
        R = project_psd(R, tol)
        R = (R + R.T) / 2.0
        R[np.diag_indices(n)] = 1.0
    return R


@dataclass
class SigmaObject:
    """Sector covariance matrix as a first-class object.

    Construct from a per-sector volatility vector ``sig`` and a
    correlation matrix ``R``; Σ is assembled as ``diag(σ) R diag(σ)``.
    Symmetry and (optionally) PSD are enforced on construction.
    """

    sig: np.ndarray                              # (n,) sector vols (std dev)
    R: np.ndarray                                # (n, n) correlation matrix
    sectors: Sequence[str] | None = None
    code: str = ""
    period: str = ""
    enforce_psd: bool = True
    tol: float = 1e-10
    Sigma: np.ndarray = field(init=False)        # populated in __post_init__

    def __post_init__(self) -> None:
        self.sig = np.asarray(self.sig, dtype=float).ravel()
        n = self.sig.shape[0]

        if self.R is None or (hasattr(self.R, "size") and self.R.size == 0):
            self.R = np.eye(n)
        self.R = clean_correlation(
            np.asarray(self.R, dtype=float),
            enforce_psd=self.enforce_psd, tol=self.tol,
        )

        if self.sectors is None:
            self.sectors = tuple(str(i) for i in range(n))
        else:
            self.sectors = tuple(str(s) for s in self.sectors)

        Sigma = (np.outer(self.sig, self.sig)) * self.R
        Sigma = (Sigma + Sigma.T) / 2.0
        self.Sigma = Sigma

    # ------------------------------------------------------------------
    # Quadratic-form queries — the hot path
    # ------------------------------------------------------------------

    def var(self, w: np.ndarray) -> float:
        """Aggregate variance ``w' Σ w``."""
        w = np.asarray(w, dtype=float).ravel()
        return float(w @ self.Sigma @ w)

    def std(self, w: np.ndarray) -> float:
        """Aggregate standard deviation ``sqrt(w' Σ w)``."""
        return float(np.sqrt(max(self.var(w), 0.0)))

    def var_no_cov(self, w: np.ndarray) -> float:
        """Diagonal-only variance ``Σ_s w_s² σ_s²``."""
        w = np.asarray(w, dtype=float).ravel()
        return float(np.nansum((w ** 2) * np.diag(self.Sigma)))

    def cov_premium_var(self, w: np.ndarray) -> float:
        """Off-diagonal contribution ``var(w) − var_no_cov(w)``."""
        return self.var(w) - self.var_no_cov(w)

    def cov_share(self, w: np.ndarray) -> float:
        """Share of total variance from cross-sector comovement."""
        tot = self.var(w)
        if tot <= 0 or np.isnan(tot):
            return float("nan")
        return self.cov_premium_var(w) / tot

    def diag_contrib(self, w: np.ndarray) -> np.ndarray:
        """Sector-by-sector diagonal contributions ``w_s² σ_s²`` (length n)."""
        w = np.asarray(w, dtype=float).ravel()
        return (w ** 2) * np.diag(self.Sigma)

    def pair_contrib(self, w: np.ndarray) -> np.ndarray:
        """Pairwise covariance-contribution matrix ``w_i w_j Σ_{ij}``,
        with the diagonal zeroed for clean accounting."""
        w = np.asarray(w, dtype=float).ravel()
        M = np.outer(w, w) * self.Sigma
        np.fill_diagonal(M, 0.0)
        return M

    # ------------------------------------------------------------------
    # Comovement diagnostics
    # ------------------------------------------------------------------

    def mean_corr(self) -> float:
        """Mean off-diagonal correlation — the simplest scalar
        summary of cross-sector comovement."""
        n = self.R.shape[0]
        if n < 2:
            return float("nan")
        iu = np.triu_indices(n, k=1)
        off = self.R[iu]
        if np.isnan(off).all():
            return float("nan")
        return float(np.nanmean(off))

    def pc1_share(self, which: str = "Sigma") -> float:
        """Fraction of variance explained by the first principal
        component of either Σ (default) or R."""
        M = self.R if which.lower() == "r" else self.Sigma
        if M is None or M.size == 0:
            return float("nan")
        M = (M + M.T) / 2.0
        ev = np.linalg.eigvalsh(M)
        ev = np.sort(ev.real)[::-1]
        tr = float(np.trace(M))
        if tr <= 0 or np.isnan(tr):
            return float("nan")
        return float(ev[0] / tr)

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------

    def summary(self, w: np.ndarray | None = None) -> pd.DataFrame:
        """One-row diagnostic table for a given weight vector.

        Columns mirror the MATLAB ``summary`` output:
        ``code, period, var_total, var_diag, var_cov, sigma_total,
        sigma_diag, cov_share, mean_corr, pc1_share_R``.
        """
        n = self.sig.shape[0]
        if w is None:
            w = np.ones(n) / n
        w = np.asarray(w, dtype=float).ravel()

        var_tot = self.var(w)
        var_diag = self.var_no_cov(w)
        var_cov = var_tot - var_diag
        return pd.DataFrame([{
            "code":          self.code,
            "period":        self.period,
            "var_total":     var_tot,
            "var_diag":      var_diag,
            "var_cov":       var_cov,
            "sigma_total":   float(np.sqrt(max(var_tot, 0.0))),
            "sigma_diag":    float(np.sqrt(max(var_diag, 0.0))),
            "cov_share":     self.cov_share(w),
            "mean_corr":     self.mean_corr(),
            "pc1_share_R":   self.pc1_share("R"),
        }])

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.sig.shape[0])


__all__ = ["SigmaObject", "project_psd", "clean_correlation"]
