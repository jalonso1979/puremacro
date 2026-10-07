"""Survey design and weighted estimation for household and person microdata.

A microdata pull is only half a result without its variance design.
Point estimates from survey weights are easy; standard errors are where
studies go wrong, because the design that produced the sample is
clustered and stratified and the public files rarely say how. Agencies
handle that by publishing *replicate weights* instead of strata and PSU
identifiers, and each survey prescribes its own variance formula over
them. :class:`SurveyDesign` records which formula applies;
:class:`MicroFrame` carries it next to the data so an estimate cannot be
separated from its standard error by accident.

VARIANCE
--------
With full-sample estimate θ and replicate estimates θ_1..θ_R::

    V = scale * Σ_r (θ_r - c)²,   c = θ        if mse
                                  c = mean(θ_r) otherwise

The presets carry each survey's documented constants:

* ACS PUMS: 80 successive-difference replicates, ``scale = 4/80``,
  centred on θ.
* SCF: 999 bootstrap replicates, ``scale = 1/998``, centred on the
  replicate mean (the variance of the 999 replicate estimates), computed
  on the first implicate only; five implicates combined with Rubin's
  rules (below). This follows the Board's *Standard Error
  Documentation*.
* CPS basic monthly: weights only. No replicate weights or design
  variables are published, so ``se`` is ``NaN`` rather than a
  simple-random-sampling number that would understate the true error.
* INEGI ENIGH: strata and PSU identifiers instead of replicates, so
  ``method="taylor"`` (below).

LINEARISATION
-------------
When a survey publishes strata and primary sampling units (PSUs) but no
replicate weights, variance is estimated by Taylor linearisation with
the with-replacement "ultimate cluster" approximation, the default of
Stata's ``svy`` and R's ``survey``. Each estimator is replaced by its
influence values z_i; with PSU totals t_hi = Σ_{k∈hi} w_k z_k::

    V = Σ_h n_h/(n_h - 1) · Σ_i (t_hi - mean_i t_hi)²

Totals use z = x; means use z = (x - θ)/Σw; quantiles use Woodruff's
method (a confidence interval for the CDF at the quantile, mapped back
through the quantile function, se = width / (2 · 1.96)). Subgroups are
domains: units outside the group keep z = 0, so every PSU still counts.
A stratum with a single PSU is centred on the grand mean of PSU totals
(``survey``'s ``lonely.psu = "adjust"``) rather than dropped.

MULTIPLE IMPUTATION
-------------------
When the design names an ``implicate`` column, each statistic is
computed per implicate and combined: estimate = mean of the m
estimates; variance = within-implicate variance + (1 + 1/m) x
between-implicate variance. The within term is the mean over implicates,
or, when ``variance_implicate`` is set, the replicate variance of that
one implicate (the SCF convention).

Everything here is NumPy / pandas and runs under Pyodide.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd

METHODS = ("sdr", "jk1", "brr", "fay", "bootstrap", "other", "taylor", "none")
_Z = 1.959963984540054   # standard normal 97.5% quantile (Woodruff interval)


@dataclass(frozen=True)
class SurveyDesign:
    """Which columns are weights and how replicate variance is formed.

    Replicate methods read ``replicate_weights``; ``method="taylor"``
    reads ``psu`` (and ``strata`` when the design is stratified)
    instead. For replicate designs, ``strata`` and ``psu`` are only
    recorded.
    """
    weight: str
    replicate_weights: tuple[str, ...] = ()
    method: str = "none"
    scale: float = 1.0
    mse: bool = True
    strata: str | None = None
    psu: str | None = None
    implicate: str | None = None
    variance_implicate: int | None = None
    note: str = ""

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"method {self.method!r} not in {METHODS}")
        if self.method == "taylor":
            if not self.psu:
                raise ValueError("method 'taylor' needs a psu column")
        elif self.method != "none" and not self.replicate_weights:
            raise ValueError(f"method {self.method!r} needs replicate_weights")

    @property
    def has_variance(self) -> bool:
        return self.method != "none"

    def columns(self) -> list[str]:
        """Every column the design refers to."""
        cols = [self.weight, *self.replicate_weights]
        cols += [c for c in (self.strata, self.psu, self.implicate) if c]
        return cols

    # ----- presets ---------------------------------------------------------
    @classmethod
    def acs_pums(cls, unit: str = "person") -> "SurveyDesign":
        """ACS PUMS: 80 successive-difference replicates (Census, *PUMS
        Accuracy of the Data*: SE = sqrt(4/80 · Σ (X_r − X)²))."""
        stem = {"person": "PWGTP", "household": "WGTP"}[unit]
        return cls(
            weight=stem,
            replicate_weights=tuple(f"{stem}{r}" for r in range(1, 81)),
            method="sdr", scale=4 / 80, mse=True,
            note="ACS PUMS successive difference replication, 80 replicates")

    @classmethod
    def scf(cls) -> "SurveyDesign":
        """SCF: 999 bootstrap replicates (wt1b_k x mm_k); sampling variance
        is the variance of the replicate estimates (scale 1/998, centred
        on their mean) on implicate 1; five implicates for Rubin's rules."""
        return cls(
            weight="wgt",
            replicate_weights=tuple(f"wt1b{r}" for r in range(1, 1000)),
            method="bootstrap", scale=1 / 998, mse=False, implicate="implicate",
            variance_implicate=1,
            note="SCF bootstrap replicates x multiplicity factors; Rubin's "
                 "rules across 5 implicates")

    @classmethod
    def enigh(cls) -> "SurveyDesign":
        """INEGI ENIGH: expansion factor ``factor``, design strata
        ``est_dis`` and PSUs ``upm``; Taylor linearisation (INEGI's own
        precision tables use the same design variables)."""
        return cls(
            weight="factor", method="taylor", strata="est_dis", psu="upm",
            note="ENIGH stratified two-stage design; Taylor linearisation, "
                 "ultimate-cluster variance")

    @classmethod
    def weights_only(cls, weight: str, note: str = "") -> "SurveyDesign":
        return cls(weight=weight, method="none", note=note)


# ---------------------------------------------------------------------------
# Weighted statistics, vectorised over [main weight, replicate weights...]
# Each takes x (n,) and W (n, K) and returns (K,) or (K, J).
# ---------------------------------------------------------------------------
def _w_total(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    return x @ W


def _w_mean(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    den = W.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (x @ W) / den


def _w_quantile(x: np.ndarray, W: np.ndarray, q: float) -> np.ndarray:
    """Inverse of the weighted empirical CDF: the smallest x whose
    cumulative weight share reaches q (each weight column separately)."""
    order = np.argsort(x, kind="stable")
    xs = x[order]
    cw = np.cumsum(W[order], axis=0)
    tot = cw[-1]
    out = np.full(W.shape[1], np.nan)
    ok = tot > 0
    if not ok.any():
        return out
    target = q * tot[ok]
    idx = (cw[:, ok] >= target - 1e-12 * np.abs(target)).argmax(axis=0)
    out[ok] = xs[idx]
    return out


@dataclass
class MicroFrame:
    """Microdata rows plus everything needed to estimate from them.

    ``data`` holds one row per unit (person or household; for multiply
    imputed surveys, one row per unit per implicate). ``query`` is the
    request that produced it with any API key removed.
    """
    data: pd.DataFrame
    design: SurveyDesign
    unit: str
    source: str
    vintage: str
    query: str = ""
    codebook: pd.DataFrame = field(default_factory=pd.DataFrame)

    def __post_init__(self):
        missing = [c for c in self.design.columns() if c not in self.data.columns]
        if missing:
            shown = missing[:5] + (["..."] if len(missing) > 5 else [])
            raise ValueError(f"design columns absent from data: {shown}")

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        return (f"MicroFrame(source={self.source!r}, vintage={self.vintage!r}, "
                f"unit={self.unit!r}, rows={len(self.data)}, "
                f"variance={self.design.method!r})")

    @property
    def variables(self) -> list[str]:
        """Analysis columns (everything that is not part of the design)."""
        design = set(self.design.columns())
        return [c for c in self.data.columns if c not in design]

    # ----- estimators ------------------------------------------------------
    def total(self, col: str, by: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Weighted population total of ``col``."""
        return self._estimate(col, by, lambda x, W: _w_total(x, W), kind="total")

    def mean(self, col: str, by: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Weighted mean of ``col``."""
        return self._estimate(col, by, _w_mean, kind="mean")

    def quantile(self, col: str, q: float,
                 by: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Weighted ``q``-quantile of ``col`` (inverse weighted CDF)."""
        if not 0 <= q <= 1:
            raise ValueError("q must lie in [0, 1]")
        return self._estimate(col, by, lambda x, W: _w_quantile(x, W, q),
                              kind=("quantile", q))

    def share(self, col: str, by: str | Sequence[str] | None = None) -> pd.DataFrame:
        """Weighted share of each category of ``col`` (within ``by``)."""
        cats = pd.Series(self.data[col].dropna().unique())
        try:
            cats = cats.sort_values()
        except TypeError:
            pass
        frames = []
        for cat in cats:
            ind = (self.data[col] == cat).astype(float).where(self.data[col].notna())
            res = self._estimate(ind, by, _w_mean, kind="mean")
            res.insert(len(res.columns) - 4, col, cat)
            frames.append(res)
        return pd.concat(frames, ignore_index=True)

    # ----- engine ----------------------------------------------------------
    def _estimate(self, col, by, stat: Callable, kind="mean") -> pd.DataFrame:
        d = self.design
        x_all = (col if isinstance(col, pd.Series) else self.data[col])
        x_all = pd.to_numeric(x_all, errors="coerce").to_numpy(dtype=float)
        by_cols = [by] if isinstance(by, str) else list(by or [])
        if d.method == "taylor":
            return self._taylor(x_all, by_cols, kind)
        wcols = [d.weight, *(d.replicate_weights if d.has_variance else ())]
        W_all = self.data[wcols].to_numpy(dtype=float, na_value=0.0)
        imp_all = (self.data[d.implicate].to_numpy() if d.implicate
                   else np.zeros(len(self.data)))

        groups = (self.data.groupby(by_cols, sort=True, dropna=False).indices
                  if by_cols else {(): np.arange(len(self.data))})
        rows = []
        for key, idx in groups.items():
            idx = np.asarray(idx)
            x = x_all[idx]
            keep = ~np.isnan(x)
            est, se, m = self._combine(x[keep], W_all[idx][keep],
                                       imp_all[idx][keep], stat)
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(by_cols, key)), "estimate": est, "se": se,
                         "n": int(keep.sum()), "implicates": m})
        return pd.DataFrame(rows, columns=[*by_cols, "estimate", "se", "n",
                                           "implicates"])

    def _taylor(self, x_all, by_cols, kind) -> pd.DataFrame:
        d = self.design
        w_all = self.data[d.weight].to_numpy(dtype=float, na_value=0.0)
        strata = (self.data[d.strata].to_numpy() if d.strata
                  else np.zeros(len(self.data)))
        psu = self.data[d.psu].to_numpy()
        groups = (self.data.groupby(by_cols, sort=True, dropna=False).indices
                  if by_cols else {(): np.arange(len(self.data))})
        rows = []
        for key, idx in groups.items():
            dom = np.zeros(len(self.data), dtype=bool)
            dom[np.asarray(idx)] = True
            dom &= ~np.isnan(x_all)
            est, se = _linearised(kind, x_all, w_all, dom, strata, psu)
            key = key if isinstance(key, tuple) else (key,)
            rows.append({**dict(zip(by_cols, key)), "estimate": est, "se": se,
                         "n": int(dom.sum()), "implicates": 1 if dom.any() else 0})
        return pd.DataFrame(rows, columns=[*by_cols, "estimate", "se", "n",
                                           "implicates"])

    def _combine(self, x, W, imp, stat):
        d = self.design
        ests, vars_ = [], []
        only = d.variance_implicate
        for i in np.unique(imp):
            sel = imp == i
            with_reps = d.has_variance and (only is None or i == only)
            theta = np.asarray(stat(x[sel], W[sel] if with_reps else W[sel][:, :1]),
                               dtype=float)
            ests.append(theta[0])
            if with_reps:
                reps = theta[1:]
                centre = theta[0] if d.mse else np.nanmean(reps)
                vars_.append(d.scale * np.nansum((reps - centre) ** 2))
        m = len(ests)
        if m == 0:
            return np.nan, np.nan, 0
        est = float(np.mean(ests))
        if not d.has_variance:
            return est, np.nan, m
        if not vars_:                       # variance implicate absent here
            return est, np.nan, m
        within = float(np.mean(vars_))
        between = float(np.var(ests, ddof=1)) if m > 1 else 0.0
        return est, float(np.sqrt(within + (1 + 1 / m) * between)), m


def _ultimate_cluster_var(u: np.ndarray, strata, psu) -> float:
    """Σ_h n_h/(n_h-1) Σ_i (t_hi - mean t_h)², t_hi = PSU totals of ``u``.

    Single-PSU strata are centred on the grand mean of PSU totals.
    """
    t = pd.DataFrame({"h": strata, "i": psu, "u": u}).groupby(
        ["h", "i"], sort=False)["u"].sum()
    n_h = t.groupby(level=0).transform("size").to_numpy()
    mean_h = t.groupby(level=0).transform("mean").to_numpy()
    tv = t.to_numpy()
    lonely = n_h == 1
    dev = np.where(lonely, tv - tv.mean(), tv - mean_h)
    factor = np.where(lonely, 1.0, n_h / np.maximum(n_h - 1, 1))
    return float(np.sum(factor * dev ** 2))


def _linearised(kind, x, w, dom, strata, psu) -> tuple[float, float]:
    """Point estimate and linearised standard error over domain ``dom``."""
    if not dom.any():
        return np.nan, np.nan
    xd, wd = x[dom], w[dom]
    wsum = wd.sum()
    z = np.zeros(len(x))
    if kind == "total":
        est = float(xd @ wd)
        z[dom] = xd
    elif kind == "mean":
        if wsum <= 0:
            return np.nan, np.nan
        est = float(xd @ wd / wsum)
        z[dom] = (xd - est) / wsum
    else:                                   # ("quantile", q): Woodruff
        q = kind[1]
        est = float(_w_quantile(xd, wd[:, None], q)[0])
        if wsum <= 0 or np.isnan(est):
            return est, np.nan
        below = (xd <= est).astype(float)
        p_hat = below @ wd / wsum
        z[dom] = (below - p_hat) / wsum
        se_p = np.sqrt(_ultimate_cluster_var(w * z, strata, psu))
        lo, hi = max(q - _Z * se_p, 0.0), min(q + _Z * se_p, 1.0)
        x_lo = float(_w_quantile(xd, wd[:, None], lo)[0])
        x_hi = float(_w_quantile(xd, wd[:, None], hi)[0])
        return est, (x_hi - x_lo) / (2 * _Z)
    return est, float(np.sqrt(_ultimate_cluster_var(w * z, strata, psu)))


def replicate_variance(theta: float, replicates: np.ndarray, *, scale: float,
                       mse: bool = True) -> float:
    """``scale · Σ (θ_r − c)²`` with ``c`` = θ (``mse``) or the replicate mean."""
    reps = np.asarray(replicates, dtype=float)
    centre = theta if mse else reps.mean()
    return float(scale * np.sum((reps - centre) ** 2))


def rubin_combine(estimates: Sequence[float],
                  variances: Sequence[float]) -> tuple[float, float]:
    """Rubin's rules: (pooled estimate, total variance)."""
    q = np.asarray(estimates, dtype=float)
    u = np.asarray(variances, dtype=float)
    m = len(q)
    b = q.var(ddof=1) if m > 1 else 0.0
    return float(q.mean()), float(u.mean() + (1 + 1 / m) * b)


__all__ = ["SurveyDesign", "MicroFrame", "METHODS", "replicate_variance",
           "rubin_combine"]
