"""Spatial autocorrelation diagnostics: Moran's I and Geary's C.

Both statistics come with the analytic moments under the normality and the
randomisation assumptions (Cliff & Ord 1981) and a permutation p-value, and
return frozen result objects with the package's presentation contract.

References
----------
Cliff, A.D. and Ord, J.K. (1981). Spatial Processes: Models and Applications. Pion.
Anselin, L. (1995). Local indicators of spatial association. Geographical Analysis 27(2).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .weights import SpatialWeights

__all__ = ["morans_i", "gearys_c", "MoranResult", "GearyResult"]


def _render(df: pd.DataFrame, fmt: str, **kwargs: Any) -> str:
    from ..reports import _df_to_latex, _df_to_markdown, _df_to_typst
    if fmt == "markdown":
        return _df_to_markdown(df, index=False, **kwargs)
    if fmt == "latex":
        return _df_to_latex(df, index=False, **kwargs)
    return _df_to_typst(df, index=False, **kwargs)


def _weights_sums(W) -> tuple[float, float, float]:
    """S0, S1, S2 of Cliff & Ord for a sparse W."""
    S0 = float(W.sum())
    Wt = W.T.tocsr()
    S1 = 0.5 * float(((W + Wt).power(2)).sum())
    row = np.asarray(W.sum(axis=1)).ravel()
    col = np.asarray(W.sum(axis=0)).ravel()
    S2 = float(np.sum((row + col) ** 2))
    return S0, S1, S2


def _coerce_x(x: Any, W: SpatialWeights) -> np.ndarray:
    if isinstance(x, pd.Series):
        missing = [u for u in W.ids if u not in x.index]
        if missing:
            raise KeyError(f"{len(missing)} unit(s) missing from x, e.g. {missing[:3]}")
        arr = x.loc[list(W.ids)].to_numpy(dtype=float)
    else:
        arr = np.asarray(x, dtype=float).ravel()
    if arr.shape[0] != W.n:
        raise ValueError(f"x has {arr.shape[0]} values, weights have {W.n} units")
    if not np.all(np.isfinite(arr)):
        raise ValueError("x contains NaN or inf")
    if W.n < 4:
        raise ValueError("spatial autocorrelation statistics need at least 4 units")
    if np.allclose(arr, arr[0]):
        raise ValueError("x is constant; the statistic is undefined")
    return arr


def _perm_p(observed: float, expected: float, sims: np.ndarray) -> float:
    n_perm = len(sims)
    if observed >= expected:
        extreme = int(np.sum(sims >= observed))
    else:
        extreme = int(np.sum(sims <= observed))
    return (extreme + 1.0) / (n_perm + 1.0)


# ---------------------------------------------------------------------------
# Result objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class MoranResult:
    """Result of :func:`morans_i`.

    Attributes
    ----------
    I : float
        Moran's I.
    expected : float
        ``-1 / (n - 1)``.
    variance_norm, variance_rand : float
        Variance under the normality and the randomisation assumption.
    z_norm, p_norm, z_rand, p_rand : float
        Standard-normal z statistics and two-sided p-values.
    p_sim : float or None
        Permutation p-value (``None`` when ``n_perm == 0``).
    n_perm : int
    n : int
    z_values : ndarray
        Centred variable, in weights order.
    lag_values : ndarray
        Spatial lag of the centred variable.
    """

    I: float
    expected: float
    variance_norm: float
    variance_rand: float
    z_norm: float
    p_norm: float
    z_rand: float
    p_rand: float
    p_sim: float | None
    n_perm: int
    n: int
    z_values: np.ndarray
    lag_values: np.ndarray

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("Moran's I", self.I), ("E[I]", self.expected),
            ("z (normality)", self.z_norm), ("p (normality)", self.p_norm),
            ("z (randomisation)", self.z_rand), ("p (randomisation)", self.p_rand),
            ("p (permutation)", self.p_sim if self.p_sim is not None else float("nan")),
            ("n", self.n), ("permutations", self.n_perm),
        ]
        return pd.DataFrame({"statistic": [r[0] for r in rows], "value": [r[1] for r in rows]})

    def summary(self) -> str:
        sim = f"{self.p_sim:.4f} ({self.n_perm} permutations)" if self.p_sim is not None else "not computed"
        return "\n".join([
            "Moran's I spatial autocorrelation",
            f"  I = {self.I:+.4f}   E[I] = {self.expected:+.4f}   n = {self.n}",
            f"  normality     : z = {self.z_norm:+.3f}, p = {self.p_norm:.4f}",
            f"  randomisation : z = {self.z_rand:+.3f}, p = {self.p_rand:.4f}",
            f"  permutation   : p = {sim}",
        ])

    def to_markdown(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, ax=None, figsize: tuple[float, float] = (5.0, 4.5)):
        """Moran scatterplot: centred variable against its spatial lag; the
        slope of the fitted line is Moran's I (for row-standardised W)."""
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        z, lz = self.z_values, self.lag_values
        ax.scatter(z, lz, s=18, color="steelblue", alpha=0.8)
        slope = float(np.dot(z, lz) / np.dot(z, z)) if np.dot(z, z) > 0 else 0.0
        grid = np.linspace(z.min(), z.max(), 50)
        ax.plot(grid, slope * grid, color="firebrick", linewidth=1.4, label=f"slope = {slope:+.3f}")
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.axvline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("x (centred)")
        ax.set_ylabel("spatial lag of x")
        ax.set_title(f"Moran scatterplot, I = {self.I:+.3f}")
        ax.legend(frameon=False)
        return fig if fig is not None else ax.get_figure()


@dataclass(frozen=True, eq=False)
class GearyResult:
    """Result of :func:`gearys_c` (fields mirror :class:`MoranResult`; the
    expectation of C is 1 and values below 1 indicate positive
    autocorrelation)."""

    C: float
    expected: float
    variance_norm: float
    variance_rand: float
    z_norm: float
    p_norm: float
    z_rand: float
    p_rand: float
    p_sim: float | None
    n_perm: int
    n: int

    def to_frame(self) -> pd.DataFrame:
        rows = [
            ("Geary's C", self.C), ("E[C]", self.expected),
            ("z (normality)", self.z_norm), ("p (normality)", self.p_norm),
            ("z (randomisation)", self.z_rand), ("p (randomisation)", self.p_rand),
            ("p (permutation)", self.p_sim if self.p_sim is not None else float("nan")),
            ("n", self.n), ("permutations", self.n_perm),
        ]
        return pd.DataFrame({"statistic": [r[0] for r in rows], "value": [r[1] for r in rows]})

    def summary(self) -> str:
        sim = f"{self.p_sim:.4f} ({self.n_perm} permutations)" if self.p_sim is not None else "not computed"
        return "\n".join([
            "Geary's C spatial autocorrelation",
            f"  C = {self.C:.4f}   E[C] = 1   n = {self.n}   (C < 1: positive autocorrelation)",
            f"  normality     : z = {self.z_norm:+.3f}, p = {self.p_norm:.4f}",
            f"  randomisation : z = {self.z_rand:+.3f}, p = {self.p_rand:.4f}",
            f"  permutation   : p = {sim}",
        ])

    def to_markdown(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "markdown", **kwargs)

    def to_latex(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "latex", **kwargs)

    def to_typst(self, **kwargs: Any) -> str:
        return _render(self.to_frame(), "typst", **kwargs)

    def plot(self, ax=None, figsize: tuple[float, float] = (5.0, 3.5)):
        """Bar of C against its expectation with the normal 95% band."""
        import matplotlib.pyplot as plt

        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        sd = float(np.sqrt(max(self.variance_rand, 0.0)))
        ax.axhspan(1 - 1.96 * sd, 1 + 1.96 * sd, color="lightgrey", alpha=0.6, label="95% band under H0")
        ax.axhline(1.0, color="grey", linewidth=1.0)
        ax.bar(["Geary's C"], [self.C], color="steelblue", width=0.4)
        ax.set_ylabel("C")
        ax.set_title(f"Geary's C = {self.C:.3f} (z = {self.z_rand:+.2f})")
        ax.legend(frameon=False)
        return fig if fig is not None else ax.get_figure()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def morans_i(x: Any, weights: SpatialWeights, *, n_perm: int = 999, seed: int | None = 0) -> MoranResult:
    """Moran's I of ``x`` under the weights ``W``.

    ``I = (n / S0) * z'Wz / z'z`` with ``z = x - mean(x)``. The normality and
    randomisation variances follow Cliff & Ord (1981); the permutation
    p-value shuffles ``x`` across units ``n_perm`` times (``n_perm=0``
    skips it).
    """
    W = weights.W
    z = _coerce_x(x, weights)
    z = z - z.mean()
    n = weights.n
    S0, S1, S2 = _weights_sums(W)
    if S0 <= 0:
        raise ValueError("weights sum to zero; no neighbours")
    zz = float(np.dot(z, z))
    lag = W @ z
    I = (n / S0) * float(np.dot(z, lag)) / zz
    EI = -1.0 / (n - 1.0)
    v_norm = (n * n * S1 - n * S2 + 3.0 * S0 * S0) / ((n * n - 1.0) * S0 * S0) - EI * EI
    b2 = n * float(np.sum(z ** 4)) / (zz * zz)
    v_rand = (
        n * ((n * n - 3.0 * n + 3.0) * S1 - n * S2 + 3.0 * S0 * S0)
        - b2 * ((n * n - n) * S1 - 2.0 * n * S2 + 6.0 * S0 * S0)
    ) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * S0 * S0) - EI * EI
    z_norm = (I - EI) / np.sqrt(v_norm) if v_norm > 0 else float("nan")
    z_rand = (I - EI) / np.sqrt(v_rand) if v_rand > 0 else float("nan")
    p_norm = float(2.0 * stats.norm.sf(abs(z_norm))) if np.isfinite(z_norm) else float("nan")
    p_rand = float(2.0 * stats.norm.sf(abs(z_rand))) if np.isfinite(z_rand) else float("nan")
    p_sim = None
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        sims = np.empty(int(n_perm))
        for b in range(int(n_perm)):
            zp = z[rng.permutation(n)]
            sims[b] = (n / S0) * float(np.dot(zp, W @ zp)) / zz
        p_sim = _perm_p(I, EI, sims)
    return MoranResult(
        I=float(I), expected=float(EI), variance_norm=float(v_norm), variance_rand=float(v_rand),
        z_norm=float(z_norm), p_norm=p_norm, z_rand=float(z_rand), p_rand=p_rand,
        p_sim=p_sim, n_perm=int(n_perm or 0), n=int(n), z_values=z, lag_values=np.asarray(lag, dtype=float),
    )


def gearys_c(x: Any, weights: SpatialWeights, *, n_perm: int = 999, seed: int | None = 0) -> GearyResult:
    """Geary's C of ``x`` under the weights ``W``:
    ``C = (n - 1) * sum_ij w_ij (x_i - x_j)^2 / (2 S0 sum_i z_i^2)``."""
    W = weights.W
    xa = _coerce_x(x, weights)
    z = xa - xa.mean()
    n = weights.n
    S0, S1, S2 = _weights_sums(W)
    if S0 <= 0:
        raise ValueError("weights sum to zero; no neighbours")
    zz = float(np.dot(z, z))
    coo = W.tocoo()

    def _c(v: np.ndarray) -> float:
        num = float(np.sum(coo.data * (v[coo.row] - v[coo.col]) ** 2))
        return (n - 1.0) * num / (2.0 * S0 * zz)

    C = _c(z)
    EC = 1.0
    v_norm = ((2.0 * S1 + S2) * (n - 1.0) - 4.0 * S0 * S0) / (2.0 * (n + 1.0) * S0 * S0)
    b2 = n * float(np.sum(z ** 4)) / (zz * zz)
    v_rand = (
        (n - 1.0) * S1 * (n * n - 3.0 * n + 3.0 - (n - 1.0) * b2)
        - 0.25 * (n - 1.0) * S2 * (n * n + 3.0 * n - 6.0 - (n * n - n + 2.0) * b2)
        + S0 * S0 * (n * n - 3.0 - (n - 1.0) ** 2 * b2)
    ) / (n * (n - 2.0) * (n - 3.0) * S0 * S0)
    z_norm = (C - EC) / np.sqrt(v_norm) if v_norm > 0 else float("nan")
    z_rand = (C - EC) / np.sqrt(v_rand) if v_rand > 0 else float("nan")
    p_norm = float(2.0 * stats.norm.sf(abs(z_norm))) if np.isfinite(z_norm) else float("nan")
    p_rand = float(2.0 * stats.norm.sf(abs(z_rand))) if np.isfinite(z_rand) else float("nan")
    p_sim = None
    if n_perm and n_perm > 0:
        rng = np.random.default_rng(seed)
        sims = np.array([_c(z[rng.permutation(n)]) for _ in range(int(n_perm))])
        p_sim = _perm_p(C, EC, sims)
    return GearyResult(
        C=float(C), expected=EC, variance_norm=float(v_norm), variance_rand=float(v_rand),
        z_norm=float(z_norm), p_norm=p_norm, z_rand=float(z_rand), p_rand=p_rand,
        p_sim=p_sim, n_perm=int(n_perm or 0), n=int(n),
    )
