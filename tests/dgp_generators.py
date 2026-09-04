"""Adversarial Data-Generating Processes (DGPs) for stress-testing puremacro estimators.

Generates stressful econometric scenarios:
  1. Ill-conditioned innovation covariance matrices (high condition number).
  2. Near-unit-root autoregressive systems (companion spectral radius rho in [0.95, 0.999]).
  3. Heavy-tailed innovations (Student-t with low degrees of freedom, skewed normals).
  4. Unsorted panels with scrambled rows and MultiIndices to test row-order invariance.
"""
from __future__ import annotations

from typing import Callable, Sequence
import numpy as np
import pandas as pd


def generate_ill_conditioned_cov(
    n: int = 3,
    cond: float = 1e4,
    seed: int = 42,
) -> np.ndarray:
    """Generate a symmetric positive-definite covariance matrix with exact condition number.

    Parameters
    ----------
    n : int
        Dimension of the covariance matrix.
    cond : float
        Condition number (ratio of largest to smallest eigenvalue >= 1.0).
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric positive-definite matrix with cond(Sigma) == cond.
    """
    rng = np.random.default_rng(seed)
    # Random orthogonal matrix via QR of standard normal
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)

    # Logarithmically spaced eigenvalues from 1.0 down to 1.0 / cond
    eigvals = np.logspace(0, -np.log10(cond), num=n)
    Sigma = Q @ np.diag(eigvals) @ Q.T
    # Enforce exact symmetry
    Sigma = 0.5 * (Sigma + Sigma.T)
    return Sigma


def generate_near_unit_root_var(
    T: int = 200,
    n: int = 2,
    rho: float = 0.98,
    p: int = 1,
    seed: int = 42,
) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Generate stable VAR(p) time series with companion spectral radius equal to rho.

    Parameters
    ----------
    T : int
        Sample length.
    n : int
        Number of endogenous variables.
    rho : float
        Target largest eigenvalue modulus (< 1.0 for covariance-stationarity).
    p : int
        Lag order.
    seed : int
        RNG seed.

    Returns
    -------
    Y : np.ndarray, shape (T, n)
        Simulated data.
    A_list : list of np.ndarray
        True coefficient matrices [A_1, ..., A_p].
    Sigma : np.ndarray, shape (n, n)
        Innovation covariance matrix.
    """
    rng = np.random.default_rng(seed)
    Sigma = generate_ill_conditioned_cov(n, cond=10.0, seed=seed)
    L = np.linalg.cholesky(Sigma)

    # Construct random companion matrix scaled to spectral radius rho
    dim = n * p
    M = rng.standard_normal((dim, dim))
    eigs = np.linalg.eigvals(M)
    max_mod = np.max(np.abs(eigs))
    M_scaled = M * (rho / max_mod)

    # Extract block coefficients A_1..A_p from first n rows
    A_list = [M_scaled[:n, l * n : (l + 1) * n] for l in range(p)]

    # Simulate with burn-in
    burn = 100
    total_T = T + burn
    eps = rng.standard_normal((total_T, n)) @ L.T
    Y_sim = np.zeros((total_T, n))
    for t in range(p, total_T):
        for l in range(p):
            Y_sim[t] += Y_sim[t - 1 - l] @ A_list[l].T
        Y_sim[t] += eps[t]

    Y = Y_sim[burn:]
    return Y, A_list, Sigma


def generate_heavy_tailed_innovations(
    T: int = 200,
    n: int = 2,
    df: float = 3.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate multivariate fat-tailed innovations from Student-t(df) distribution.

    Parameters
    ----------
    T : int
        Number of time steps.
    n : int
        Dimension.
    df : float
        Degrees of freedom (low df => heavier tails).
    seed : int
        RNG seed.

    Returns
    -------
    np.ndarray, shape (T, n)
        Innovations with fat tails.
    """
    rng = np.random.default_rng(seed)
    # Student-t variate: normal / sqrt(chi2(df)/df)
    z = rng.standard_normal((T, n))
    u = rng.chisquare(df, size=(T, 1))
    scale = np.sqrt(u / df)
    return z / scale


def generate_unsorted_panel(
    n_units: int = 8,
    n_periods: int = 60,
    seed: int = 42,
    shuffle: bool = True,
) -> pd.DataFrame:
    """Generate an econometric panel dataset, optionally with scrambled row order.

    DGP:
        y_{i,t} = alpha_i + 0.6 y_{i,t-1} + 0.7 x_{i,t} + e_{i,t}

    Parameters
    ----------
    n_units : int
        Number of panel entities.
    n_periods : int
        Number of time periods per entity.
    seed : int
        RNG seed.
    shuffle : bool
        If True, shuffles rows completely.

    Returns
    -------
    pd.DataFrame
        MultiIndexed DataFrame with levels ["code", "date"].
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        code = f"unit_{i:02d}"
        alpha_i = rng.standard_normal() * 0.5
        x = rng.standard_normal(n_periods)
        e = rng.standard_normal(n_periods) * 0.5
        z = 0.8 * x + rng.standard_normal(n_periods) * 0.3  # Instrument
        y = np.zeros(n_periods)
        for t in range(1, n_periods):
            y[t] = alpha_i + 0.6 * y[t - 1] + 0.7 * x[t] + e[t]
        for t in range(n_periods):
            rows.append({"code": code, "date": t, "y": y[t], "x": x[t], "z": z[t]})

    df = pd.DataFrame(rows).set_index(["code", "date"])
    if shuffle:
        df = df.sample(frac=1.0, random_state=seed)
    return df
