"""Frozen-dataclass result objects for puremacro.garch."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GARCH11Result:
    """Result of :func:`puremacro.garch.fit.garch11_fit`.

    Attributes
    ----------
    omega : float
        Constant in the GARCH(1,1) recursion.
    alpha : float
        ARCH coefficient (loading on lagged squared shock).
    beta : float
        GARCH coefficient (loading on lagged conditional variance).
    sigma : pd.Series
        Conditional volatility σ_t.
    loglik : float
        Log-likelihood at the MLE.
    converged : bool
        Optimiser convergence flag.
    persistence : float
        ``α + β``.

    References
    ----------
    Bollerslev, T. (1986). Generalized autoregressive conditional
        heteroskedasticity. Journal of Econometrics 31(3), 307-327.
    """

    omega: float
    alpha: float
    beta: float
    sigma: pd.Series
    loglik: float
    converged: bool
    persistence: float

    def summary(self) -> str:
        return (
            f"GARCH(1,1) fit\n"
            f"  ω                 : {self.omega:.6f}\n"
            f"  α                 : {self.alpha:.4f}\n"
            f"  β                 : {self.beta:.4f}\n"
            f"  persistence (α+β) : {self.persistence:.4f}\n"
            f"  log-lik           : {self.loglik:+.4f}\n"
            f"  converged         : {self.converged}\n"
        )


@dataclass(frozen=True)
class DCCResult:
    """Result of :func:`puremacro.garch.dcc.dcc_fit`.

    Attributes
    ----------
    a : float
        DCC innovation parameter.
    b : float
        DCC persistence parameter.
    Qbar : ndarray, shape (n, n)
        Unconditional standardised-residual covariance.
    sigma : pd.DataFrame, shape (T, n)
        Per-asset conditional volatility.
    R : ndarray, shape (T, n, n)
        Conditional correlation path.
    H : ndarray, shape (T, n, n)
        Conditional covariance path.
    garch_params : list of dict
        Per-asset GARCH(1,1) parameters and diagnostics.
    loglik : float
        DCC-stage log-likelihood.
    converged : bool
        Optimiser convergence flag.

    References
    ----------
    Engle, R. (2002). Dynamic conditional correlation: a simple class of
        multivariate generalized autoregressive conditional
        heteroskedasticity models. JBES 20(3), 339-350.
    """

    a: float
    b: float
    Qbar: np.ndarray
    sigma: pd.DataFrame
    R: np.ndarray
    H: np.ndarray
    garch_params: list
    loglik: float
    converged: bool

    def summary(self) -> str:
        T, n = self.sigma.shape
        return (
            f"DCC(1,1) fit\n"
            f"  series (n)        : {n}\n"
            f"  observations (T)  : {T}\n"
            f"  a                 : {self.a:.4f}\n"
            f"  b                 : {self.b:.4f}\n"
            f"  a + b             : {self.a + self.b:.4f}\n"
            f"  log-lik (DCC)     : {self.loglik:+.4f}\n"
            f"  converged         : {self.converged}\n"
        )
