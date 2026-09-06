"""Spatial weights: sparse ``W`` matrices built from neighbour lists, coordinates
or economic flows.

Everything here runs on numpy, scipy.sparse and pandas only (no geometry
stack), so it works under Pyodide. Units are identified by an ``ids`` tuple;
``lag(x)`` aligns pandas inputs on those labels.

References
----------
Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer.
LeSage, J. and Pace, R.K. (2009). Introduction to Spatial Econometrics. CRC.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

EARTH_RADIUS_KM = 6371.0088

__all__ = [
    "SpatialWeights",
    "contiguity_weights",
    "knn_weights",
    "distance_weights",
    "economic_weights",
    "haversine_km",
    "pairwise_distances",
]


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------
def haversine_km(coords_a: Any, coords_b: Any | None = None) -> np.ndarray:
    """Great-circle distances in kilometres between rows of ``(n, 2)`` arrays
    of ``[latitude, longitude]`` in degrees. Returns an ``(n_a, n_b)`` matrix
    (``coords_b`` defaults to ``coords_a``)."""
    a = np.asarray(coords_a, dtype=float)
    b = a if coords_b is None else np.asarray(coords_b, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or b.ndim != 2 or b.shape[1] != 2:
        raise ValueError("haversine_km expects (n, 2) arrays of [lat, lon] in degrees")
    if np.any(np.abs(a[:, 0]) > 90) or np.any(np.abs(b[:, 0]) > 90):
        raise ValueError("latitude must lie in [-90, 90] degrees (did you pass [lon, lat]?)")
    lat1 = np.radians(a[:, 0])[:, None]
    lon1 = np.radians(a[:, 1])[:, None]
    lat2 = np.radians(b[:, 0])[None, :]
    lon2 = np.radians(b[:, 1])[None, :]
    h = np.sin((lat2 - lat1) / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(h, 0.0, 1.0)))


def pairwise_distances(coords: Any, metric: str = "haversine") -> np.ndarray:
    """Dense ``(n, n)`` distance matrix. ``metric`` is ``'haversine'`` (km,
    coordinates are ``[lat, lon]`` degrees) or ``'euclidean'`` (same units
    as the coordinates)."""
    a = np.asarray(coords, dtype=float)
    if a.ndim != 2 or a.shape[1] < 2:
        raise ValueError("coords must be an (n, d) array with d >= 2")
    m = metric.lower()
    if m == "haversine":
        return haversine_km(a[:, :2])
    if m == "euclidean":
        diff = a[:, None, :] - a[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))
    raise ValueError(f"metric must be 'haversine' or 'euclidean', got {metric!r}")


def _coerce_coords(coords: Any, ids: Sequence[Any] | None) -> tuple[np.ndarray, tuple]:
    """Accept a DataFrame (index = ids, first two columns = coordinates), a
    Series-of-tuples, a mapping id -> (lat, lon), or an array with ``ids``."""
    if isinstance(coords, pd.DataFrame):
        arr = coords.iloc[:, :2].to_numpy(dtype=float)
        labels = tuple(coords.index) if ids is None else tuple(ids)
    elif isinstance(coords, Mapping):
        labels = tuple(coords.keys()) if ids is None else tuple(ids)
        arr = np.array([coords[k] for k in labels], dtype=float)
    else:
        arr = np.asarray(coords, dtype=float)
        labels = tuple(range(len(arr))) if ids is None else tuple(ids)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("coords must provide two coordinates per unit")
    if not np.all(np.isfinite(arr[:, :2])):
        raise ValueError("coords must be finite")
    if len(labels) != len(arr):
        raise ValueError(f"ids has {len(labels)} entries but coords has {len(arr)} rows")
    if len(set(labels)) != len(labels):
        raise ValueError("unit ids must be unique")
    return arr[:, :2], labels


def _row_standardize(W: sp.csr_matrix) -> sp.csr_matrix:
    rs = np.asarray(W.sum(axis=1)).ravel()
    inv = np.where(rs > 0, 1.0 / np.where(rs > 0, rs, 1.0), 0.0)
    return sp.diags(inv) @ W


# ---------------------------------------------------------------------------
# The weights object
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SpatialWeights:
    """A sparse spatial weights matrix with unit labels.

    Attributes
    ----------
    W : scipy.sparse.csr_matrix, shape (n, n)
        Weights; ``W[i, j]`` is the weight unit ``i`` puts on unit ``j``.
        The diagonal is zero.
    ids : tuple
        Unit labels in row/column order.
    kind : str
        How the matrix was built (``'contiguity'``, ``'knn'``, ``'distance'``,
        ``'economic'`` or ``'custom'``).
    row_standardized : bool
        ``True`` when every non-island row sums to one.
    """

    W: sp.csr_matrix
    ids: tuple
    kind: str = "custom"
    row_standardized: bool = False

    def __post_init__(self) -> None:
        W = sp.csr_matrix(self.W, dtype=float)
        if W.shape[0] != W.shape[1]:
            raise ValueError(f"W must be square, got shape {W.shape}")
        if len(self.ids) != W.shape[0]:
            raise ValueError(f"ids has {len(self.ids)} labels for a {W.shape[0]}x{W.shape[0]} matrix")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("unit ids must be unique")
        if W.nnz and W.data.min() < 0:
            raise ValueError("spatial weights must be non-negative")
        if W.diagonal().any():
            W = W.tolil()
            W.setdiag(0.0)
            W = W.tocsr()
        W.eliminate_zeros()
        object.__setattr__(self, "W", W)
        object.__setattr__(self, "ids", tuple(self.ids))

    # -- basic properties ---------------------------------------------------
    @property
    def n(self) -> int:
        return self.W.shape[0]

    @property
    def n_neighbors(self) -> np.ndarray:
        """Number of non-zero weights in each row."""
        return np.diff(self.W.indptr)

    @property
    def n_islands(self) -> int:
        """Units with no neighbours."""
        return int(np.sum(self.n_neighbors == 0))

    @property
    def islands(self) -> tuple:
        return tuple(self.ids[i] for i in np.flatnonzero(self.n_neighbors == 0))

    @property
    def is_symmetric(self) -> bool:
        return bool(abs(self.W - self.W.T).max() < 1e-12) if self.W.nnz else True

    @property
    def s0(self) -> float:
        """Sum of all weights."""
        return float(self.W.sum())

    def row_sums(self) -> np.ndarray:
        return np.asarray(self.W.sum(axis=1)).ravel()

    # -- transformations ----------------------------------------------------
    def standardize(self) -> "SpatialWeights":
        """Row-standardised copy (rows sum to one; islands stay zero)."""
        return SpatialWeights(_row_standardize(self.W), self.ids, self.kind, True)

    def binary(self) -> "SpatialWeights":
        """Copy with every non-zero weight set to one."""
        B = self.W.copy()
        B.data[:] = 1.0
        return SpatialWeights(B, self.ids, self.kind, False)

    def to_dense(self) -> np.ndarray:
        return self.W.toarray()

    def index_of(self, unit: Any) -> int:
        try:
            return self.ids.index(unit)
        except ValueError as exc:
            raise KeyError(f"unit {unit!r} is not in the weights (ids are {self.ids[:5]}...)") from exc

    def neighbors(self, unit: Any) -> dict:
        """``{neighbour id: weight}`` for one unit."""
        i = self.index_of(unit)
        row = self.W.getrow(i)
        return {self.ids[j]: float(w) for j, w in zip(row.indices, row.data)}

    # -- the spatial lag ----------------------------------------------------
    def lag(self, x: Any) -> Any:
        """Spatial lag ``W @ x``.

        ``x`` may be an array of shape ``(n,)`` or ``(n, k)`` in ``ids`` order,
        or a pandas Series/DataFrame indexed by the unit ids (aligned by
        label; a missing unit raises). Pandas in, pandas out.
        """
        if isinstance(x, (pd.Series, pd.DataFrame)):
            missing = [u for u in self.ids if u not in x.index]
            if missing:
                raise KeyError(f"lag: {len(missing)} unit(s) missing from the input index, e.g. {missing[:3]}")
            xa = x.loc[list(self.ids)]
            out = self.W @ xa.to_numpy(dtype=float)
            if isinstance(x, pd.Series):
                return pd.Series(out, index=pd.Index(self.ids, name=x.index.name), name=x.name)
            return pd.DataFrame(out, index=pd.Index(self.ids, name=x.index.name), columns=x.columns)
        arr = np.asarray(x, dtype=float)
        if arr.shape[0] != self.n:
            raise ValueError(f"lag: x has {arr.shape[0]} rows, weights have {self.n} units")
        return self.W @ arr

    # -- presentation -------------------------------------------------------
    def to_frame(self) -> pd.DataFrame:
        """Edge list with columns ``source``, ``target``, ``weight``."""
        coo = self.W.tocoo()
        return pd.DataFrame({
            "source": [self.ids[i] for i in coo.row],
            "target": [self.ids[j] for j in coo.col],
            "weight": coo.data,
        })

    def neighbor_table(self) -> pd.DataFrame:
        return pd.DataFrame({
            "unit": list(self.ids),
            "n_neighbors": self.n_neighbors,
            "row_sum": self.row_sums(),
        })

    def summary(self) -> str:
        nn = self.n_neighbors
        lines = [
            f"SpatialWeights ({self.kind}) — {self.n} units, {self.W.nnz} non-zero weights",
            f"  row-standardised : {self.row_standardized}",
            f"  symmetric        : {self.is_symmetric}",
            f"  neighbours/unit  : min {int(nn.min()) if self.n else 0}, mean {float(nn.mean()) if self.n else 0.0:.2f}, max {int(nn.max()) if self.n else 0}",
            f"  islands          : {self.n_islands}" + (f" {list(self.islands)[:5]}" if self.n_islands else ""),
            f"  S0 (sum of weights): {self.s0:.4f}",
        ]
        return "\n".join(lines)

    def to_markdown(self, **kwargs: Any) -> str:
        from ..reports import _df_to_markdown
        return _df_to_markdown(self.neighbor_table(), index=False, **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        from ..reports import _df_to_latex
        return _df_to_latex(self.neighbor_table(), index=False, **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        from ..reports import _df_to_typst
        return _df_to_typst(self.neighbor_table(), index=False, **kwargs)

    def plot(self, figsize: tuple[float, float] = (9.0, 3.6)):
        """Sparsity pattern of ``W`` and the neighbour-count histogram."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].spy(self.W, markersize=max(0.5, 60.0 / max(self.n, 1)))
        axes[0].set_title(f"W ({self.kind}, n={self.n})")
        nn = self.n_neighbors
        axes[1].hist(nn, bins=range(0, int(nn.max()) + 2) if self.n else 1, color="steelblue", edgecolor="white")
        axes[1].set_xlabel("neighbours per unit")
        axes[1].set_ylabel("units")
        axes[1].set_title("neighbour counts")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def contiguity_weights(
    neighbors: Mapping[Any, Iterable[Any]],
    *,
    ids: Sequence[Any] | None = None,
    symmetric: bool = True,
    row_standardize: bool = True,
) -> SpatialWeights:
    """Binary contiguity weights from a neighbour mapping ``{id: [ids...]}``.

    Unit order follows the mapping's keys, so a plain array passed to
    ``morans_i`` or ``W.lag`` in that order is read correctly; units that
    appear only as neighbours are appended after the keys. ``symmetric=True``
    makes every listed pair mutual. Row-standardised by default.
    """
    if ids is None:
        seen: list = list(neighbors.keys())
        seen_set = set(seen)
        for vs in neighbors.values():
            for v in vs:
                if v not in seen_set:
                    seen.append(v)
                    seen_set.add(v)
        labels = tuple(seen)
    else:
        labels = tuple(ids)
    pos = {u: i for i, u in enumerate(labels)}
    rows, cols = [], []
    for k, vs in neighbors.items():
        if k not in pos:
            raise KeyError(f"unit {k!r} is not in ids")
        for v in vs:
            if v not in pos:
                raise KeyError(f"neighbour {v!r} of {k!r} is not in ids")
            if v == k:
                continue
            rows.append(pos[k])
            cols.append(pos[v])
            if symmetric:
                rows.append(pos[v])
                cols.append(pos[k])
    n = len(labels)
    W = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    W.data[:] = 1.0  # duplicates summed above -> back to binary
    out = SpatialWeights(W, labels, "contiguity", False)
    return out.standardize() if row_standardize else out


def knn_weights(
    coords: Any,
    k: int,
    *,
    ids: Sequence[Any] | None = None,
    metric: str = "haversine",
    row_standardize: bool = True,
) -> SpatialWeights:
    """k-nearest-neighbour weights (binary before standardisation).

    ``coords`` is a DataFrame indexed by unit id with ``[lat, lon]`` (or
    ``[x, y]`` with ``metric='euclidean'``) as its first two columns, a
    mapping ``id -> (lat, lon)``, or an ``(n, 2)`` array with ``ids``.
    """
    arr, labels = _coerce_coords(coords, ids)
    n = len(labels)
    if not (1 <= k < n):
        raise ValueError(f"k must satisfy 1 <= k < n (n={n}), got {k}")
    D = pairwise_distances(arr, metric)
    np.fill_diagonal(D, np.inf)
    nbr = np.argpartition(D, k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    W = sp.csr_matrix((np.ones(n * k), (rows, nbr.ravel())), shape=(n, n))
    out = SpatialWeights(W, labels, "knn", False)
    return out.standardize() if row_standardize else out


def distance_weights(
    coords: Any,
    cutoff: float,
    *,
    ids: Sequence[Any] | None = None,
    metric: str = "haversine",
    decay: str = "inverse",
    power: float = 1.0,
    bandwidth: float | None = None,
    row_standardize: bool = True,
) -> SpatialWeights:
    """Distance-band weights: pairs closer than ``cutoff`` (km for
    ``'haversine'``) are neighbours with weight ``1/d**power``
    (``decay='inverse'``), ``1`` (``'uniform'``) or ``exp(-(d/bandwidth)^2/2)``
    (``'gaussian'``, ``bandwidth`` defaults to ``cutoff``). Units with no
    neighbour inside the cutoff are islands (a warning names them).
    """
    arr, labels = _coerce_coords(coords, ids)
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    D = pairwise_distances(arr, metric)
    n = len(labels)
    mask = (D <= cutoff) & ~np.eye(n, dtype=bool)
    d = decay.lower()
    if d == "inverse":
        with np.errstate(divide="ignore"):
            vals = np.where(mask & (D > 0), 1.0 / np.where(D > 0, D, 1.0) ** power, 0.0)
        dup = mask & (D == 0)
        if dup.any():
            warnings.warn(
                "distance_weights: some units share the same coordinates; their inverse-distance "
                "weight is set to the largest finite weight in the matrix",
                RuntimeWarning, stacklevel=2,
            )
            vals[dup] = vals[vals > 0].max() if (vals > 0).any() else 1.0
    elif d == "uniform":
        vals = mask.astype(float)
    elif d == "gaussian":
        bw = cutoff if bandwidth is None else float(bandwidth)
        vals = np.where(mask, np.exp(-0.5 * (D / bw) ** 2), 0.0)
    else:
        raise ValueError(f"decay must be 'inverse', 'uniform' or 'gaussian', got {decay!r}")
    W = sp.csr_matrix(vals)
    out = SpatialWeights(W, labels, "distance", False)
    if out.n_islands:
        warnings.warn(
            f"distance_weights: {out.n_islands} unit(s) have no neighbour within {cutoff} "
            f"({list(out.islands)[:5]}); they are islands with a zero row",
            RuntimeWarning, stacklevel=2,
        )
    return out.standardize() if row_standardize else out


def economic_weights(
    flows: Any,
    *,
    ids: Sequence[Any] | None = None,
    row_standardize: bool = True,
) -> SpatialWeights:
    """Weights from an ``(n, n)`` matrix of economic flows (trade, commuting,
    input-output linkages): ``W[i, j]`` is the share of ``i``'s flows that go
    to ``j``. The diagonal is dropped; negative flows raise."""
    if isinstance(flows, pd.DataFrame):
        labels = tuple(flows.index) if ids is None else tuple(ids)
        if ids is None and list(flows.columns) != list(flows.index):
            flows = flows.loc[list(labels), list(labels)]
        arr = flows.to_numpy(dtype=float)
    else:
        arr = np.asarray(flows, dtype=float)
        labels = tuple(range(arr.shape[0])) if ids is None else tuple(ids)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("flows must be a square (n, n) matrix")
    if np.any(arr < 0):
        raise ValueError("economic flows must be non-negative")
    arr = arr.copy()
    np.fill_diagonal(arr, 0.0)
    out = SpatialWeights(sp.csr_matrix(arr), labels, "economic", False)
    return out.standardize() if row_standardize else out
