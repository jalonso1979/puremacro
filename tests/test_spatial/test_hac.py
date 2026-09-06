"""Conley spatial HAC: brute-force equality and its limiting cases."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from puremacro.inference.dk import driscoll_kraay
from puremacro.spatial import conley_cov, conley_se, kernel_matrix, spatial_hac_panel_cov, spatial_hac_panel_meat


def _design(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0.0, 10.0, (n, 2))
    X = np.column_stack([np.ones(n), rng.standard_normal(n), rng.standard_normal(n)])
    e = rng.standard_normal(n)
    return coords, X, e


def _brute(X, e, coords, cutoff, kernel):
    n = len(e)
    D = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    if kernel == "bartlett":
        K = np.where(D <= cutoff, 1 - D / cutoff, 0.0)
    else:
        K = (D <= cutoff).astype(float)
    np.fill_diagonal(K, 1.0)
    U = X * e[:, None]
    S = np.zeros((X.shape[1], X.shape[1]))
    for i in range(n):
        for j in range(n):
            S += K[i, j] * np.outer(U[i], U[j])
    XtXi = np.linalg.inv(X.T @ X)
    return XtXi @ S @ XtXi


@pytest.mark.parametrize("kernel", ["bartlett", "uniform"])
def test_conley_equals_brute_force(kernel):
    coords, X, e = _design(40)
    cov = conley_cov(X, e, coords, 3.0, kernel=kernel, metric="euclidean")
    np.testing.assert_allclose(cov, _brute(X, e, coords, 3.0, kernel), atol=1e-12)
    np.testing.assert_allclose(conley_se(X, e, coords, 3.0, kernel=kernel, metric="euclidean"), np.sqrt(np.diag(cov)))


def test_zero_cutoff_is_hc0():
    coords, X, e = _design(50)
    XtXi = np.linalg.inv(X.T @ X)
    hc0 = XtXi @ (X * e[:, None] ** 2).T @ X @ XtXi
    np.testing.assert_allclose(conley_cov(X, e, coords, 0.0, metric="euclidean"), hc0, atol=1e-12)


def test_uniform_kernel_with_far_apart_clusters_is_cluster_robust():
    rng = np.random.default_rng(3)
    n_per, G = 12, 4
    centres = np.array([[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]])
    coords = np.vstack([c + rng.uniform(-1.0, 1.0, (n_per, 2)) for c in centres])
    cluster = np.repeat(np.arange(G), n_per)
    n = len(cluster)
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    e = rng.standard_normal(n)
    cov = conley_cov(X, e, coords, 5.0, kernel="uniform", metric="euclidean")
    U = X * e[:, None]
    S = sum(np.outer(U[cluster == g].sum(0), U[cluster == g].sum(0)) for g in range(G))
    XtXi = np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov, XtXi @ S @ XtXi, atol=1e-12)


def test_kernel_matrix_validation():
    D = np.array([[0.0, 1.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        kernel_matrix(D, -1.0)
    with pytest.raises(ValueError):
        kernel_matrix(D, 1.0, kernel="epanechnikov")
    with pytest.raises(ValueError, match="one row per observation"):
        conley_cov(np.ones((3, 1)), np.ones(3), np.zeros((2, 2)), 1.0)


def _panel(n_e=6, T=15, seed=1):
    rng = np.random.default_rng(seed)
    ents = np.repeat(np.arange(n_e), T)
    times = np.tile(np.arange(T), n_e)
    X = np.column_stack([np.ones(n_e * T), rng.standard_normal(n_e * T)])
    e = rng.standard_normal(n_e * T)
    coords = pd.DataFrame(rng.uniform(0, 10, (n_e, 2)), index=np.arange(n_e), columns=["x", "y"])
    return X, e, coords, ents, times


def test_panel_meat_with_zero_time_lags_is_per_period_conley():
    X, e, coords, ents, times = _panel()
    S = spatial_hac_panel_meat(X, e, coords, ents, times, 4.0, 0, metric="euclidean")
    S_ref = np.zeros((2, 2))
    for t in np.unique(times):
        m = times == t
        XtXi = np.linalg.inv(X[m].T @ X[m])
        cov_t = conley_cov(X[m], e[m], coords.loc[ents[m]].to_numpy(), 4.0, metric="euclidean")
        S_ref += np.linalg.inv(XtXi) @ cov_t @ np.linalg.inv(XtXi)  # back out the meat
    np.testing.assert_allclose(S, S_ref, atol=1e-9)


def test_panel_meat_with_everything_inside_the_cutoff_is_driscoll_kraay():
    X, e, coords, ents, times = _panel()
    for L in (0, 2, 4):
        S = spatial_hac_panel_meat(X, e, coords, ents, times, 1e9, L, kernel="uniform", metric="euclidean")
        S_dk = driscoll_kraay(X * e[:, None], times, lags=L)
        np.testing.assert_allclose(S, S_dk, atol=1e-10)
    cov = spatial_hac_panel_cov(X, e, coords, ents, times, 1e9, 2, kernel="uniform", metric="euclidean")
    XtXi = np.linalg.inv(X.T @ X)
    np.testing.assert_allclose(cov, XtXi @ driscoll_kraay(X * e[:, None], times, lags=2) @ XtXi, atol=1e-10)


def test_panel_meat_validation():
    X, e, coords, ents, times = _panel()
    with pytest.raises(KeyError):
        spatial_hac_panel_meat(X, e, coords.iloc[:-1], ents, times, 4.0, 0, metric="euclidean")
    with pytest.raises(ValueError):
        spatial_hac_panel_meat(X, e, coords, ents, times, 4.0, -1, metric="euclidean")
    with pytest.raises(ValueError):
        spatial_hac_panel_meat(X[:-1], e, coords, ents, times, 4.0, 0, metric="euclidean")
