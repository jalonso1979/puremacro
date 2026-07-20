"""Python port of SigmaObject.m — variance/covariance premium for rolling panels.

Var(w'g) = w' Σ w, where Σ = diag(σ) R diag(σ).

Defaults to PSD-enforced Σ (eigenvalue clipping at tol=1e-10) — matches the
MATLAB implementation in `MAV/SigmaObject.m`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class SigmaObject:
    sigma: np.ndarray          # 1-D length N
    R: np.ndarray              # N × N correlation matrix
    labels: Sequence[str]
    country: str = ""
    period: str = ""
    psd_enforce: bool = True
    psd_tol: float = 1e-10
    Sigma: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.sigma = np.asarray(self.sigma, dtype=float)
        self.R = np.asarray(self.R, dtype=float)
        # symmetrize
        self.R = 0.5 * (self.R + self.R.T)
        self.Sigma = np.outer(self.sigma, self.sigma) * self.R
        if self.psd_enforce:
            self.Sigma = self._clip_psd(self.Sigma, self.psd_tol)

    @staticmethod
    def _clip_psd(M: np.ndarray, tol: float) -> np.ndarray:
        vals, vecs = np.linalg.eigh(M)
        vals_clipped = np.clip(vals, tol, None)
        return vecs @ np.diag(vals_clipped) @ vecs.T

    # --- quadratic forms ---
    def var(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        return float(w @ self.Sigma @ w)

    def var_no_cov(self, w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        return float(np.sum((w * self.sigma) ** 2))

    def cov_premium_var(self, w: np.ndarray) -> float:
        v = self.var(w)
        v_diag = self.var_no_cov(w)
        if v_diag == 0:
            return 0.0
        return (v - v_diag) / v_diag

    def mean_corr(self) -> float:
        off = self.R[np.triu_indices_from(self.R, k=1)]
        return float(off.mean()) if off.size else 0.0

    def diag_contrib(self, w: np.ndarray) -> np.ndarray:
        w = np.asarray(w, dtype=float)
        return (w * self.sigma) ** 2
