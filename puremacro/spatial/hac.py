"""Spatial and space-time HAC covariance (Conley 1999).

``conley_cov`` is the cross-section estimator: ``(X'X)^-1 S (X'X)^-1`` with
``S = sum_i sum_j K(d_ij) u_i u_j'`` and ``u_i = X_i e_i``. ``K`` is the
Bartlett kernel ``1 - d/cutoff`` (Conley's default) or the uniform kernel
inside the cutoff; ``K(0) = 1`` so ``cutoff = 0`` gives the HC0 estimator.

``spatial_hac_panel_cov`` adds a Bartlett kernel in time (Hsiang 2010):
``S = sum_{t,s} w(|t - s|) sum_{i,j} K(d_ij) u_it u_js'`` with
``w(l) = 1 - l/(L+1)``. With ``time_lags = 0`` it is Conley period by
period; with a cutoff larger than every distance it is the Driscoll-Kraay
(1998) meat. The distance matrix is dense: memory is O(n^2) in the number of
units, which is fine for regional panels.

References
----------
Conley, T.G. (1999). GMM estimation with cross sectional dependence. J. Econometrics 92(1).
Hsiang, S.M. (2010). Temperatures and cyclones strongly associated with economic
    production in the Caribbean and Central America. PNAS 107(35).
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from .weights import _coerce_coords, pairwise_distances

__all__ = ["conley_cov", "conley_se", "spatial_hac_panel_cov", "spatial_hac_panel_meat", "kernel_matrix"]


def kernel_matrix(D: np.ndarray, cutoff: float, kernel: str = "bartlett") -> np.ndarray:
    """Spatial kernel weights ``K(d_ij)`` for a distance matrix ``D``."""
    k = kernel.lower()
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    if cutoff == 0:
        return np.eye(D.shape[0])
    inside = D <= cutoff
    if k == "bartlett":
        K = np.where(inside, 1.0 - D / cutoff, 0.0)
    elif k == "uniform":
        K = inside.astype(float)
    else:
        raise ValueError(f"kernel must be 'bartlett' or 'uniform', got {kernel!r}")
    np.fill_diagonal(K, 1.0)
    return K


def _xtx_inv(X: np.ndarray, name: str) -> np.ndarray:
    from ..inference.hac import inv_xtx
    return inv_xtx(X, name=name)


def conley_cov(
    X: Any,
    resid: Any,
    coords: Any,
    cutoff_km: float,
    *,
    kernel: str = "bartlett",
    metric: str = "haversine",
) -> np.ndarray:
    """Conley (1999) spatial HAC covariance of OLS coefficients.

    Parameters
    ----------
    X : (n, k) regressor matrix (include the constant if used).
    resid : (n,) OLS residuals.
    coords : (n, 2) ``[lat, lon]`` (or ``[x, y]`` with ``metric='euclidean'``),
        one row per observation, in the same order as ``X``.
    cutoff_km : float
        Distance beyond which observations are treated as independent.
    kernel : {'bartlett', 'uniform'}
    """
    X = np.asarray(X, dtype=float)
    e = np.asarray(resid, dtype=float).ravel()
    if X.ndim != 2 or X.shape[0] != e.shape[0]:
        raise ValueError("X must be (n, k) and resid (n,) with matching n")
    arr = np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != X.shape[0]:
        raise ValueError("coords must have one row per observation")
    D = pairwise_distances(arr[:, :2], metric)
    K = kernel_matrix(D, float(cutoff_km), kernel)
    U = X * e[:, None]
    S = U.T @ K @ U
    XtX_inv = _xtx_inv(X, "conley_cov")
    return XtX_inv @ S @ XtX_inv


def conley_se(X: Any, resid: Any, coords: Any, cutoff_km: float, *, kernel: str = "bartlett", metric: str = "haversine") -> np.ndarray:
    """Square root of the diagonal of :func:`conley_cov`."""
    return np.sqrt(np.diag(conley_cov(X, resid, coords, cutoff_km, kernel=kernel, metric=metric)))


def _entity_coords(coords: Any, entities: np.ndarray) -> np.ndarray:
    """Coordinates for each unique entity key, from a DataFrame / mapping /
    array-with-ids as accepted by the weights builders."""
    if isinstance(coords, pd.DataFrame):
        missing = [u for u in entities if u not in coords.index]
        if missing:
            raise KeyError(f"coords is missing {len(missing)} entit(ies), e.g. {missing[:3]}")
        return coords.loc[list(entities)].iloc[:, :2].to_numpy(dtype=float)
    if isinstance(coords, Mapping):
        missing = [u for u in entities if u not in coords]
        if missing:
            raise KeyError(f"coords is missing {len(missing)} entit(ies), e.g. {missing[:3]}")
        return np.array([coords[u] for u in entities], dtype=float)[:, :2]
    arr, ids = _coerce_coords(coords, None)
    pos = {u: i for i, u in enumerate(ids)}
    missing = [u for u in entities if u not in pos]
    if missing:
        raise KeyError(
            f"coords is missing {len(missing)} entit(ies), e.g. {missing[:3]}; pass a DataFrame "
            "indexed by entity id or a mapping id -> (lat, lon)"
        )
    return arr[[pos[u] for u in entities]]


def spatial_hac_panel_meat(
    X: Any,
    resid: Any,
    coords: Any,
    entity_keys: Any,
    time_keys: Any,
    cutoff_km: float,
    time_lags: int = 0,
    *,
    kernel: str = "bartlett",
    metric: str = "haversine",
) -> np.ndarray:
    """The ``S`` matrix of the space-time HAC sandwich (see module docstring).

    ``coords`` gives one coordinate pair per *entity* (DataFrame indexed by
    entity id, mapping, or array aligned with ``sorted(unique(entity_keys))``);
    ``entity_keys`` / ``time_keys`` label each row of ``X``.
    """
    X = np.asarray(X, dtype=float)
    e = np.asarray(resid, dtype=float).ravel()
    ent = np.asarray(entity_keys)
    tim = np.asarray(time_keys)
    if not (X.shape[0] == e.shape[0] == ent.shape[0] == tim.shape[0]):
        raise ValueError("X, resid, entity_keys and time_keys must have the same number of rows")
    if time_lags < 0:
        raise ValueError("time_lags must be non-negative")
    entities, ent_idx = np.unique(ent, return_inverse=True)
    periods, t_idx = np.unique(tim, return_inverse=True)
    n_e, T, k = len(entities), len(periods), X.shape[1]
    C = _entity_coords(coords, entities)
    K = kernel_matrix(pairwise_distances(C, metric), float(cutoff_km), kernel)
    # scores stacked as (T, n_e, k); absent entity-periods stay zero
    U = np.zeros((T, n_e, k))
    U[t_idx, ent_idx, :] = X * e[:, None]
    KU = np.einsum("ij,tjk->tik", K, U)          # K applied within each period
    S = np.zeros((k, k))
    for t in range(T):
        S += U[t].T @ KU[t]
    L = int(time_lags)
    for ell in range(1, L + 1):
        if ell >= T:
            break
        w = 1.0 - ell / (L + 1.0)
        G = np.zeros((k, k))
        for t in range(ell, T):
            G += U[t].T @ KU[t - ell]
        S += w * (G + G.T)
    return S


def spatial_hac_panel_cov(
    X: Any,
    resid: Any,
    coords: Any,
    entity_keys: Any,
    time_keys: Any,
    cutoff_km: float,
    time_lags: int = 0,
    *,
    kernel: str = "bartlett",
    metric: str = "haversine",
) -> np.ndarray:
    """Space-time HAC covariance ``(X'X)^-1 S (X'X)^-1`` of OLS coefficients
    on a panel (Conley in space, Bartlett with bandwidth ``time_lags`` in time)."""
    S = spatial_hac_panel_meat(X, resid, coords, entity_keys, time_keys, cutoff_km, time_lags, kernel=kernel, metric=metric)
    XtX_inv = _xtx_inv(np.asarray(X, dtype=float), "spatial_hac_panel_cov")
    return XtX_inv @ S @ XtX_inv
