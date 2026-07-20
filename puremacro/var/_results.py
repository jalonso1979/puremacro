"""Frozen-dataclass result objects for puremacro.var.* estimators."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VarEstimateResult:
    """Result of :func:`puremacro.var.estimate.estimate_var`.

    Attributes
    ----------
    A_list : list of ndarray, length p
        VAR coefficient matrices A_1, ..., A_p, each shape (n, n).
    c : ndarray, shape (n,)
        Constant term.
    Sigma : ndarray, shape (n, n)
        Reduced-form residual covariance.
    resid : ndarray, shape (T - p, n)
        Reduced-form residuals.
    X : ndarray, shape (T - p, 1 + n*p)
        Design matrix (constant + p lags).
    """

    A_list: list[np.ndarray]
    c: np.ndarray
    Sigma: np.ndarray
    resid: np.ndarray
    X: np.ndarray

    def __iter__(self):
        """Support legacy `A_list, c, Sigma, resid, X = estimate_var(...)` unpack."""
        yield self.A_list
        yield self.c
        yield self.Sigma
        yield self.resid
        yield self.X

    def __len__(self):
        """Return 5 so that `len(result) == 5` guards in existing tests pass."""
        return 5

    def __getitem__(self, index: int):
        """Support legacy `result[i]` indexing: 0=A_list, 1=c, 2=Sigma, 3=resid, 4=X."""
        fields = (self.A_list, self.c, self.Sigma, self.resid, self.X)
        return fields[index]

    def summary(self) -> str:
        p = len(self.A_list)
        n = self.Sigma.shape[0]
        T_eff = self.resid.shape[0]
        return (
            f"VAR estimate\n"
            f"  variables (n)     : {n}\n"
            f"  lag order (p)     : {p}\n"
            f"  effective T       : {T_eff}\n"
            f"  Σ trace           : {float(np.trace(self.Sigma)):.4f}\n"
        )
