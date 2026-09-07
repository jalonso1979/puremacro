"""Spatial difference-in-differences: exposure rings around treated units.

A treatment at some locations spills onto nearby untreated ones. That breaks
SUTVA twice over: the direct effect is contaminated because the control group
is partly treated, and the spillover itself is an object of interest that the
canonical two-way-FE DiD cannot see. The standard fix (Clarke 2017; Berg,
Reisinger & Streitz 2021; Butts 2023) partitions the *untreated* units into
distance RINGS around the treated ones, gives each ring its own coefficient,
and keeps only the units beyond the outermost ring as the comparison group.

Estimating equation (common timing, ``Post_t = 1{t >= t0}``)::

    y_it = mu_i + tau_t + sum_{r=0}^{R} delta_r * Ring^r_it + eps_it

``Ring^0_it = 1`` when unit ``i`` is itself treated at ``t``; ``Ring^r_it = 1``
when ``i`` is untreated at ``t`` and its distance to the nearest treated unit
falls in band ``r``. The omitted category is "untreated and beyond
``rings[-1]``". Ring 0 is *absorbing*: a treated unit next to another treated
unit stays in ring 0, so ``delta_0`` is the **total effect on the treated**
(its own direct effect plus any spillover it receives from other treated
units), not the pure direct ATT. ``delta_1..delta_R`` are the spillovers.

Bands. With ``edges = rings``, band 1 is the closed interval
``[0, edges[1]]`` and band ``r >= 2`` is the half-open interval
``(edges[r-1], edges[r]]``; ``np.searchsorted(edges[1:], d, side='left') + 1``
is what delivers exactly that. A distance beyond ``edges[-1]`` — and the
``+inf`` of "nothing nearby has been treated yet" — lands in the control code
``R + 1``.

Timing. Under staggered adoption the ring indicator is time-varying: by
default (``ring_timing='already_treated'``) ``Ring^r_it`` uses the distance to
the nearest *already*-treated unit at ``t``, so every ring indicator is
identically zero before any nearby treatment happens and the pre-period stays
a clean baseline. ``ring_timing='ever_treated'`` switches the ring on at the
nearest treated unit's own adoption date; it is the right choice when
anticipation is the mechanism, and it mechanically contaminates the
pre-period otherwise.

The bias-correction reading. Let ``M`` be the two-way within projection. The
naive DiD that pools every untreated unit into the control group is exactly
the short regression, so by Frisch-Waugh (1933)

    delta_naive = delta_0 + sum_{r=1}^{R} theta_r delta_r,
    theta_r = (M Ring^0)' (M Ring^r) / (M Ring^0)' (M Ring^0).

Ring indicators are mutually exclusive and switch on together, so
``theta_r < 0`` in every realistic design: a *positive* spillover biases the
naive estimate *downward*. :func:`spatial_did` reports both numbers and the
contamination term ``sum_r theta_r delta_r`` that separates them; the identity
holds to machine precision because both come from one within projection, and
only inside that one projection — it says nothing about a naive regression run
on a different sample.

Inference is Conley (1999) space-time HAC by default (``cov_type='conley'``,
via :func:`puremacro.spatial.hac.spatial_hac_panel_meat`). Cluster-by-unit is
available and under-covers here: rings are a deterministic function of
geography, so regressor and error are both spatially correlated, and
clustering by unit throws away every off-diagonal ``K(d_ij) u_i u_j'`` term.
The Conley standard error is itself valid only under *increasing-domain*
asymptotics (the region expands, the mixing field decays, the cutoff grows
slowly). It is not valid under infill asymptotics, and it says little when the
treated units form a handful of spatially separated experiments —
``n_treated_clusters`` is the honest diagnostic and ``summary()`` prints it.

What the outer-ring test can and cannot say
-------------------------------------------
``delta_R`` is identified as (outermost ring) minus (the beyond-ring
controls). A spillover that is *common* to the outermost ring and the controls
shifts both by the same amount and leaves ``delta_R`` exactly zero. So
``outer_ring_contrast_bound`` bounds a **difference** between the outermost
ring and the controls, never the level of the spillover out there. The field
name, the docstring and ``summary()`` all say "contrast" for that reason. The
only real defence is sensitivity to widening ``rings[-1]``.

Joint Wald tests are only as good as the units behind them
----------------------------------------------------------
Every joint statistic (``no_spillover``, ``equal_rings``, ``dynamics_ring*``
and the pre-trend rows) is a quadratic form in a *robust* covariance. When a
ring is carried by a handful of units that covariance behaves like a sample
covariance with that many degrees of freedom, and the chi-square reference
distribution is badly wrong. Under G6 of this release's build rules a badly
sized statistic does not ship, so **these rows are not reported at all in that
regime**: any joint restriction that leans on a ring carried by fewer than
``_MIN_UNITS_FOR_JOINT`` (= 10) units comes back with ``stat = p = NaN`` and
``underpowered=True`` in ``tests`` / ``pretrend``, and ``spatial_did`` warns
naming the rings. The per-coefficient ring table is unaffected: a
single-coefficient Wald stays correctly sized however small the ring is.

The measurements behind that rule, all on seeded null designs (no spillover,
no pre-trend, two-way-FE noise only, ``cov_type='cluster'``, normal critical
values, nominal 0.10):

* **Ungated small design** (90 units, 9 treated, T = 10, 500-unit box, 200
  replications; ring sizes 9 / 12 / 7 / 44): joint pre-trend **0.515**,
  per-ring pre-trend 0.212, ``no_spillover`` 0.140, ``equal_rings`` 0.160.
  Per-coefficient inference in the same runs is fine — ``delta_0`` rejects
  0.080 and covers 0.955 at a nominal 0.95. This is the regime the gate now
  refuses to report.
* **Pooled by ring size** over 870 designs (n = 90..400, 9..45 treated), the
  per-ring pre-trend rejects 0.285 with fewer than 10 units in the ring and
  0.085-0.144 above; ``no_spillover`` 0.231 below 10 and 0.079-0.110 above.
  Ten units is where the distortion goes away, which is why the threshold sits
  there.
* **Designs that clear the gate** (200 units, 22 treated, two seed blocks of
  80): ``no_spillover`` 0.10 and 0.05, per-ring pre-trend 0.17 and 0.10.

So the joint rows that *do* ship are approximately correctly sized, with the
per-ring pre-trend the least reliable of them (0.10-0.17 at a nominal 0.10 with
every ring past the gate) — read that one as indicative. The all-rings joint
pre-trend was worse than any of them and is not reported at all; see
"Deliberately omitted". ``n_treated_clusters``, not ``n_obs``, is the sample
size that governs all of this, and ``summary()`` prints it.

Deliberately omitted
--------------------
* **The all-rings joint pre-trend Wald (G6).** ``pretrend`` reports one row per
  ring and no ``ring = -1`` joint row. Twelve restrictions read off a robust
  covariance carried by a few dozen treated units is not a chi-square: on
  seeded nulls at a nominal 0.10 it rejected 0.515 (90 units / 9 treated),
  0.20 and 0.18 (two 80-replication blocks of 200 units / 22 treated, with
  *every* ring clearing the size gate below), 0.135 (200 / 20), 0.095
  (300 / 30) and 0.087 (400 / 45). It is over-sized wherever the design is
  small enough for the question to matter, so it does not ship. The per-ring
  rows carry the same information three restrictions at a time and are
  approximately sized; a Bonferroni or Holm adjustment across them is the
  honest joint statement.
* **``estimator='cs'`` (a per-ring Callaway-Sant'Anna loop).** It needs one
  scalar treat time per unit, so it cannot represent a unit that sits in ring 2
  at ``t = 5`` and ring 1 at ``t = 8``; it produces no cross-ring covariance,
  which leaves the contamination decomposition and every joint test undefined;
  and its panel bootstrap ignores exactly the spatial correlation this module
  exists to handle. Run :func:`puremacro.did.callaway_santanna` yourself on the
  per-ring sub-panel if you want those point estimates.
* **The headline never follows ``event_study_estimator``.** ``ring_table``,
  ``direct_effect``, ``naive_att``, ``contamination``, ``total_effect`` and all
  four Wald tests come from the static two-way-FE fit — ``static_estimator`` is
  fixed at ``'twfe'`` and ``summary()`` says so on every call, with the
  Goodman-Bacon (2021) / de Chaisemartin-D'Haultfoeuille (2020) negative-weight
  caveat printed whenever there is more than one cohort. Only the event study
  and ``att_by_ring_es`` follow ``event_study_estimator``. The Frisch-Waugh
  identity holds inside the static projection and nowhere else, which is why
  the headline cannot be allowed to drift.
* **Choropleths and any geometry stack.** Pure numpy / scipy / pandas, so the
  module runs under Pyodide. ``matplotlib`` is imported inside ``plot()``.
* **Anticipation windows, doughnut designs, continuous exposure measures and
  ring-boundary selection.** Ring cut points are a researcher degree of
  freedom; fix them ex ante on substantive grounds and report sensitivity.
  :meth:`RingAssignment.plot` exists to help you cut where the distance
  distribution is thin.
* **A t-distribution for the critical values.** Under spatial dependence the
  effective degrees of freedom are not ``n - k``; the module uses standard
  normal critical values and says so.

References
----------
Butts, K. (2023). Difference-in-differences estimation with spatial
    spillovers. arXiv:2105.03737.
Clarke, D. (2017). Estimating difference-in-differences in the presence of
    spillovers. MPRA Paper 81604.
Berg, T., Reisinger, M. and Streitz, D. (2021). Spillover effects in empirical
    corporate finance. Journal of Financial Economics 142(3), 1109-1127.
Delgado, M.S. and Florax, R.J.G.M. (2015). Difference-in-differences
    techniques for spatial data: local autocorrelation and spatial
    interaction. Economics Letters 137, 123-126.
Huber, M. and Steinmayr, A. (2021). A framework for separating individual-
    level treatment effects from spillover effects. Journal of Business &
    Economic Statistics 39(2), 422-436.
Verbitsky-Savitz, N. and Raudenbush, S.W. (2012). Causal inference under
    interference in spatial settings. Epidemiologic Methods 1(1), 107-130.
Conley, T.G. (1999). GMM estimation with cross sectional dependence.
    Journal of Econometrics 92(1), 1-45.
Hsiang, S.M. (2010). Temperatures and cyclones strongly associated with
    economic production in the Caribbean and Central America. PNAS 107(35),
    15367-15372.
Frisch, R. and Waugh, F.V. (1933). Partial time regressions as compared with
    individual trends. Econometrica 1(4), 387-401.
Sun, L. and Abraham, S. (2021). Estimating dynamic treatment effects in event
    studies with heterogeneous treatment effects. Journal of Econometrics
    225(2), 175-199.
Goodman-Bacon, A. (2021). Difference-in-differences with variation in
    treatment timing. Journal of Econometrics 225(2), 254-277.
de Chaisemartin, C. and D'Haultfoeuille, X. (2020). Two-way fixed effects
    estimators with heterogeneous treatment effects. American Economic Review
    110(9), 2964-2996.
Cameron, A.C. and Miller, D.L. (2015). A practitioner's guide to cluster-
    robust inference. Journal of Human Resources 50(2), 317-372.
Driscoll, J.C. and Kraay, A.C. (1998). Consistent covariance matrix estimation
    with spatially dependent panel data. Review of Economics and Statistics
    80(4), 549-560.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse as sp
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.stats import chi2, norm

from .._linalg import inv_xtx
from ..lp._panel_helpers import as_panel_index, two_way_fe_within
from ..spatial.weights import SpatialWeights, _coerce_coords, haversine_km

__all__ = [
    "spatial_did",
    "exposure_rings",
    "contiguity_rings",
    "SpatialDiDResult",
    "RingAssignment",
]

_RING_TIMINGS = ("already_treated", "ever_treated")
_METRICS = ("haversine", "euclidean")
_COV_TYPES = ("conley", "cluster")
_KERNELS = ("bartlett", "uniform")
_PSD_ADJUST = ("none", "warn", "clip")
_ES_ESTIMATORS = ("auto", "twfe", "sunab")

#: Sentinel for "the caller did not supply this keyword". ``spatial_did`` needs
#: to tell `metric='haversine'` (explicit, and meaningless on some routes) from
#: `metric` left alone, because G3 forbids silently ignoring a keyword.
_UNSET = object()

#: Below this many units in a ring, the joint Wald statistics that touch it are
#: badly over-sized (the robust meat behaves like a sample covariance with that
#: many degrees of freedom). ``spatial_did`` reports those rows as
#: ``stat = p = NaN`` with ``underpowered=True`` and warns; it does not print a
#: confident p-value from a reference distribution it knows is wrong.
#:
#: The threshold is measured, not guessed. Pooling 870 seeded null designs
#: (n in 90..400, 9..45 treated, two-way-FE noise only, ``cov_type='cluster'``)
#: and bucketing every per-ring pre-trend Wald by the number of units carrying
#: that ring, the rejection rate at a nominal 0.10 is 0.285 below 10 units and
#: 0.096 / 0.144 / 0.118 / 0.085 / 0.099 in the [10,15) / [15,20) / [20,30) /
#: [30,50) / [50,inf) buckets. ``no_spillover`` and ``equal_rings`` break at the
#: same place (0.231 and 0.228 below 10 units, 0.07-0.11 above).
_MIN_UNITS_FOR_JOINT = 10

#: ``outer_ring_contrast_share`` is reported as NaN when the direct effect is
#: smaller than this, because the ratio is then numerically meaningless.
_SHARE_FLOOR = 1e-8


# ---------------------------------------------------------------------------
# Small shared helpers (candidates for hoisting; see the module report)
# ---------------------------------------------------------------------------
def _render(df: pd.DataFrame, fmt: str, **kwargs: Any) -> str:
    """Dispatch a frame to one of the three ``reports`` renderers."""
    from ..reports import _df_to_latex, _df_to_markdown, _df_to_typst

    if fmt == "markdown":
        return _df_to_markdown(df, index=False, **kwargs)
    if fmt == "latex":
        return _df_to_latex(df, index=False, **kwargs)
    return _df_to_typst(df, index=False, **kwargs)


def _wald(
    beta: np.ndarray,
    vcov: np.ndarray,
    R: np.ndarray,
    q: np.ndarray | None = None,
    *,
    name: str = "spatial_did",
    rcond: float = 1e-10,
) -> tuple[float, int, float]:
    """Wald statistic ``(Rb - q)' pinv(R V R') (Rb - q)`` and its p-value.

    The pseudo-inverse (rather than a plain solve) is deliberate: the
    two-dimensional Bartlett kernel is not positive definite, so ``V`` can be
    indefinite and ``R V R'`` can be numerically rank deficient. The reported
    degrees of freedom are the *numerical rank* of ``R V R'`` at tolerance
    ``rcond`` relative to its largest singular value, not the row count of
    ``R``.

    Parameters
    ----------
    beta : (k,) ndarray
        Coefficient vector.
    vcov : (k, k) ndarray
        Its covariance.
    R : (m, k) ndarray
        Restriction matrix.
    q : (m,) ndarray, optional
        Restriction values; zeros by default.
    name : str
        Caller name, used in error messages.
    rcond : float, default 1e-10
        Relative singular-value cutoff for both the pseudo-inverse and the
        rank decision. It is relative to the largest singular value, so a
        uniform rescaling of the outcome does not change the reported rank.

    Returns
    -------
    (stat, df, p) : tuple of (float, int, float)
        ``(nan, 0, nan)`` when the effective rank is zero.

    Raises
    ------
    ValueError
        If ``R`` is not two-dimensional with ``k`` columns, or ``q`` has the
        wrong length.
    """
    b = np.asarray(beta, dtype=float).ravel()
    V = np.asarray(vcov, dtype=float)
    R = np.atleast_2d(np.asarray(R, dtype=float))
    if R.ndim != 2 or R.shape[1] != b.shape[0]:
        raise ValueError(
            f"{name}: the restriction matrix must be (m, {b.shape[0]}), got {R.shape}")
    qq = np.zeros(R.shape[0]) if q is None else np.asarray(q, dtype=float).ravel()
    if qq.shape[0] != R.shape[0]:
        raise ValueError(
            f"{name}: q has {qq.shape[0]} entries for {R.shape[0]} restrictions")
    d = R @ b - qq
    A = R @ V @ R.T
    A = 0.5 * (A + A.T)
    s = np.linalg.svd(A, compute_uv=False)
    if s.size == 0 or not np.isfinite(s).all() or s[0] <= 0:
        return float("nan"), 0, float("nan")
    rank = int(np.sum(s > rcond * s[0]))
    if rank == 0:
        return float("nan"), 0, float("nan")
    stat = float(d @ np.linalg.pinv(A, rcond=rcond, hermitian=True) @ d)
    if not np.isfinite(stat) or stat < 0:
        return float("nan"), rank, float("nan")
    return stat, rank, float(chi2.sf(stat, rank))


def _distance_block(a: np.ndarray, b: np.ndarray, metric: str) -> np.ndarray:
    """``(n_a, n_b)`` distances between two coordinate blocks.

    The rectangular form matters: ring construction only ever needs distances
    from all ``N`` units to the ``N_treated`` treated ones, which is a fraction
    of the memory a full ``(N, N)`` matrix would take.
    """
    if metric == "haversine":
        return haversine_km(a[:, :2], b[:, :2])
    if metric == "euclidean":
        diff = a[:, None, :2] - b[None, :, :2]
        return np.sqrt(np.sum(diff ** 2, axis=-1))
    raise ValueError(f"metric must be one of {_METRICS}, got {metric!r}")


def _treated_cluster_count(coords: np.ndarray, cutoff: float, metric: str) -> int:
    """Connected components of the treated-unit graph at radius ``cutoff``.

    This is the count of spatially independent treatment experiments, and it
    governs how well the normal approximation behind the Conley critical values
    works. Components over *all* units are useless for a contiguous region
    (they are 1 by construction); restricting to the treated units is what
    makes the diagnostic informative.
    """
    n = coords.shape[0]
    if n == 0:
        return 0
    if cutoff <= 0:
        return int(n)
    D = _distance_block(coords, coords, metric)
    A = sp.csr_matrix((D <= float(cutoff)).astype(np.int8))
    ncomp, _ = connected_components(A, directed=False)
    return int(ncomp)


def _fmt_edge(x: float) -> str:
    if not np.isfinite(x):
        return "inf"
    if float(x) == int(x):
        return str(int(x))
    return f"{float(x):g}"


def _default_labels(edges: Sequence[float], unit: str) -> tuple[str, ...]:
    """``('treated', '0-25 km', ..., '>100 km')`` from the band edges.

    ``unit='unknown'`` (the ``ring_col`` route, where the codes are the
    caller's and no distance was ever computed) drops the unit suffix rather
    than inventing kilometres for a column that may hold hop counts.
    """
    e = [float(v) for v in edges]
    if unit == "hops":
        bands = [f"{_fmt_edge(e[r])} hop" + ("" if e[r] == 1 else "s")
                 for r in range(1, len(e))]
        return ("treated", *bands, f">{_fmt_edge(e[-1])} hops")
    suffix = "" if unit == "unknown" else f" {unit}"
    bands = [f"{_fmt_edge(e[r - 1])}-{_fmt_edge(e[r])}{suffix}"
             for r in range(1, len(e))]
    return ("treated", *bands, f">{_fmt_edge(e[-1])}{suffix}")


def _validate_edges(rings: Sequence[float], func: str) -> np.ndarray:
    """Band cut points: strictly increasing, finite, starting at 0.0."""
    try:
        edges = np.asarray(list(rings), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{func}: `rings` must be a sequence of floats, got {rings!r}") from exc
    if edges.ndim != 1 or edges.size < 2:
        raise ValueError(
            f"{func}: `rings` needs at least two cut points (got {edges.size}); "
            "the first must be 0.0 and each later one is the outer edge of a band")
    if not np.all(np.isfinite(edges)):
        raise ValueError(f"{func}: `rings` must be finite, got {list(rings)!r}")
    if edges[0] != 0.0:
        raise ValueError(
            f"{func}: `rings` must start at 0.0 (prepend 0.0), got {list(rings)!r}")
    if np.any(np.diff(edges) <= 0):
        raise ValueError(
            f"{func}: `rings` must be strictly increasing, got {list(rings)!r}")
    if np.any(edges < 0):
        raise ValueError(f"{func}: `rings` must be non-negative, got {list(rings)!r}")
    return edges


def _coerce_treat_time(treat_time: Any, ids: Sequence[Any], func: str) -> list:
    """Per-unit first-treatment period, aligned to ``ids``.

    Accepts a Series indexed by unit id, a mapping, or an array in ``ids``
    order. ``NaN`` / ``None`` / ``inf`` all mean never-treated.
    """
    if isinstance(treat_time, pd.Series):
        missing = [u for u in ids if u not in treat_time.index]
        if missing:
            raise KeyError(
                f"{func}: treat_time is missing {len(missing)} unit(s), "
                f"e.g. {missing[:3]}")
        return list(treat_time.loc[list(ids)].to_numpy())
    if isinstance(treat_time, Mapping):
        missing = [u for u in ids if u not in treat_time]
        if missing:
            raise KeyError(
                f"{func}: treat_time is missing {len(missing)} unit(s), "
                f"e.g. {missing[:3]}")
        return [treat_time[u] for u in ids]
    arr = list(np.asarray(treat_time, dtype=object).ravel())
    if len(arr) != len(ids):
        raise ValueError(
            f"{func}: treat_time has {len(arr)} entries for {len(ids)} units; "
            "pass a Series/mapping keyed by unit id, or an array in `ids` order")
    return arr


def _is_never(v: Any) -> bool:
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        return False
    if isinstance(v, (int, float, np.integer, np.floating)):
        return not np.isfinite(float(v))
    return False


def _entry_positions(periods: pd.Index, values: Sequence[Any], func: str) -> np.ndarray:
    """Position in ``periods`` of each unit's first treated period.

    Returns ``len(periods)`` for a never-treated unit and for a treat time
    later than every observed period (both are "never treated in sample").
    """
    n = len(values)
    pos = np.full(n, len(periods), dtype=int)
    live = np.array([not _is_never(v) for v in values], dtype=bool)
    if live.any():
        vals = [values[i] for i in np.flatnonzero(live)]
        try:
            p = np.asarray(periods.searchsorted(np.asarray(vals), side="left"), dtype=int)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{func}: treat_time values are not comparable with the period "
                f"labels (periods are {list(periods[:3])}..., treat_time has "
                f"{vals[:3]}...)") from exc
        pos[live] = np.clip(p, 0, len(periods))
    return pos


def _running_min_by_cohort(
    D: np.ndarray,
    treated_cohort: np.ndarray,
    treated_global: np.ndarray,
    n_cohorts: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative nearest-treated distance after each cohort has switched on.

    ``D`` is ``(N, N_treated)``. Returns ``(M, A)`` of shape
    ``(n_cohorts, N)``: ``M[k]`` is the distance to the nearest unit treated by
    cohort ``k`` inclusive, ``A[k]`` the index (in the full unit ordering) of
    that unit. Cost is ``O(N * N_treated)`` in total, not ``O(N * N_treated * T)``.

    Ties: the update is strict, so the earliest cohort wins, and within a
    cohort ``np.argmin`` picks the first unit in ``ids`` order. The *distance*
    is a minimum and so is never ambiguous; only the reported identity is.
    """
    n = D.shape[0]
    M = np.full(n, np.inf)
    A = np.full(n, -1, dtype=int)
    out_m = np.empty((n_cohorts, n))
    out_a = np.empty((n_cohorts, n), dtype=int)
    for k in range(n_cohorts):
        cols = np.flatnonzero(treated_cohort == k)
        if cols.size:
            block = D[:, cols]
            j = np.argmin(block, axis=1)
            v = block[np.arange(n), j]
            upd = v < M
            A[upd] = treated_global[cols[j[upd]]]
            M[upd] = v[upd]
        out_m[k] = M
        out_a[k] = A
    return out_m, out_a


def _event_clock(code: np.ndarray, R: int) -> tuple:
    """Entry period, entry ring, event clock and switcher flags from codes.

    ``code`` is ``(n_units, n_periods)`` with the control code ``R + 1`` and
    the excluded sentinel ``-1``. Entry ``a_i`` is the first period of any
    exposure (treatment included), ``entry_ring_i`` the code there and
    ``event_time_it = t - a_i``. Units never exposed carry ``NaN`` event times
    and are the omitted comparison group throughout.
    """
    n, T = code.shape
    exposed = (code >= 0) & (code <= R)
    has_entry = exposed.any(axis=1)
    a_idx = np.where(has_entry, exposed.argmax(axis=1), -1)
    entry_ring = np.where(has_entry, code[np.arange(n), np.maximum(a_idx, 0)], -1)
    ev = np.where(has_entry[:, None],
                  np.arange(T)[None, :] - a_idx[:, None], np.nan)
    switch = np.zeros(n, dtype=bool)
    for i in range(n):
        if not has_entry[i]:
            continue
        tail = code[i, a_idx[i]:]
        switch[i] = bool(np.any(tail[(tail >= 0) & (tail <= R)] != entry_ring[i]))
    return has_entry, a_idx, entry_ring, ev, switch


def _codes_from_distance(D: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Band code ``1..R+1`` from a distance array.

    ``side='left'`` gives ``[0, edges[1]]`` for band 1 and
    ``(edges[r-1], edges[r]]`` for ``r >= 2``; ``+inf`` maps to ``R + 1``.
    """
    return np.searchsorted(edges[1:], D, side="left") + 1


# ---------------------------------------------------------------------------
# RingAssignment
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class RingAssignment:
    """The (unit, time) exposure-ring assignment behind a spatial DiD.

    Attributes
    ----------
    frame : pd.DataFrame
        Long, one row per ``(unit, time)``. Columns ``unit``, ``time``,
        ``distance`` (float, kilometres or hops; ``+inf`` before anything
        nearby is treated), ``ring`` (int, ``0..R+1``), ``ring_label`` (str),
        ``entry_ring`` (nullable ``Int64``; ``<NA>`` for units never exposed),
        ``entry_time`` (the period ``a_i``; ``NaN``/``NaT`` for those units),
        ``event_time`` (float, ``NaN`` for them), ``nearest_treated``
        (unit id or ``None``) and ``treated`` (bool, the *time-varying*
        ``1{t >= g_i}``, not an ever-treated flag —
        :meth:`unit_frame` carries the ever-treated version).
    counts : pd.DataFrame
        One row per ring ``0..R+1``: ``ring``, ``label``, ``lo_edge``,
        ``hi_edge``, ``n_units``, ``n_unit_periods``. ``n_unit_periods``
        partitions all ``N * T`` rows and therefore sums to ``len(frame)``;
        ``n_units`` counts units *ever* in the ring and therefore does **not**
        sum to ``N`` when the rings are time-varying.
    ids : tuple
        Unit labels in coordinate order.
    times : tuple
        Period labels, sorted.
    edges : tuple of float
        Band cut points (hop counts for :func:`contiguity_rings`).
    labels : tuple of str
        Length ``R + 2``; the last entry labels the control group.
    n_rings : int
        ``R``, the number of spillover bands (excludes ring 0 and the control).
    ring_timing : str
        ``'already_treated'`` or ``'ever_treated'``.
    metric : str
        ``'haversine'``, ``'euclidean'`` or ``'graph'``.
    distance_unit : str
        ``'km'`` for haversine, ``'units'`` for euclidean, ``'hops'`` for graph.
        The ``lo_edge`` / ``hi_edge`` columns are in these units.
    is_static : bool
        ``True`` when no unit ever changes its exposed ring code, i.e.
        ``n_switchers == 0``. :meth:`unit_frame` needs this.
    n_switchers : int
        Units whose ring code changes after they first become exposed.
    n_treated : int
        Units treated at some observed period.
    n_control : int
        Units never inside any ring at any period.
    n_islands : int
        Units with no finite distance to any treated unit (the graph route);
        0 for the coordinate routes.
    """

    frame: pd.DataFrame
    counts: pd.DataFrame
    ids: tuple
    times: tuple
    edges: tuple
    labels: tuple
    n_rings: int
    ring_timing: str
    metric: str
    distance_unit: str
    is_static: bool
    n_switchers: int
    n_treated: int
    n_control: int
    n_islands: int

    # -- presentation -------------------------------------------------------
    def to_frame(self, which: str = "counts") -> pd.DataFrame:
        """``counts`` (default) or the long ``assignment`` frame."""
        if which == "counts":
            return self.counts.copy()
        if which == "assignment":
            return self.frame.copy()
        raise ValueError(
            f"RingAssignment.to_frame: which must be 'counts' or 'assignment', "
            f"got {which!r}")

    def unit_frame(self) -> pd.DataFrame:
        """One row per unit: distance, ring, entry ring, entry time.

        ``treated`` is the **ever-treated** flag. ``frame['treated']`` is the
        time-varying ``1{t >= g_i}`` indicator, so collapsing it on the first
        observed period would call every staggered-adoption unit untreated
        while the same row says ``ring = 0``; this column is
        ``frame.groupby('unit')['treated'].any()`` instead.

        ``distance`` is the distance to the nearest treated unit **at the
        period of exposure entry**. A unit that is never exposed has no entry,
        and then the value reported is its distance in the *last* observed
        period — the finite nearest-treated distance of the comparison group,
        which is exactly what you need to defend the outermost cut point. It
        is ``+inf`` only when the unit really is unreachable (a graph island,
        or a panel where nothing is ever treated nearby).

        Raises
        ------
        ValueError
            When ``is_static`` is False — a unit that walks inward through two
            rings has no single ring code, so the collapse would be a lie.
        """
        if not self.is_static:
            raise ValueError(
                "RingAssignment.unit_frame: the ring code is time-varying "
                f"({self.n_switchers} unit(s) change ring); use `.frame` instead")
        f = self.frame
        exposed = f["ring"] <= self.n_rings
        first = f[exposed].groupby("unit", sort=False).head(1)
        last = f.groupby("unit", sort=False).tail(1).set_index("unit")
        out = f.groupby("unit", sort=False).head(1)[["unit"]].copy()
        out["treated"] = out["unit"].map(
            f.groupby("unit", sort=False)["treated"].any()).astype(bool)
        keep = first.set_index("unit")
        out["distance"] = out["unit"].map(keep["distance"])
        out["distance"] = out["distance"].where(
            out["distance"].notna(), out["unit"].map(last["distance"]))
        out["ring"] = out["unit"].map(keep["ring"]).astype("Int64")
        out["entry_ring"] = out["unit"].map(keep["entry_ring"]).astype("Int64")
        out["entry_time"] = out["unit"].map(keep["entry_time"])
        out["ring_label"] = out["ring"].map(
            {r: self.labels[r] for r in range(self.n_rings + 2)})
        out.loc[out["ring"].isna(), "ring"] = self.n_rings + 1
        out["ring"] = out["ring"].astype("Int64")
        out["ring_label"] = out["ring_label"].fillna(self.labels[-1])
        return out.reset_index(drop=True)

    def summary(self) -> str:
        """Human-readable ring census."""
        u = self.distance_unit
        head = (f"RingAssignment ({self.metric}, {self.ring_timing}) — "
                f"{len(self.ids)} units x {len(self.times)} periods")
        lines = [head,
                 "  edges          : "
                 + " / ".join(_fmt_edge(e) for e in self.edges)
                 + f" {u}  ({self.n_rings} spillover ring"
                 + ("s" if self.n_rings != 1 else "") + ")",
                 f"  treated        : {self.n_treated} units",
                 ]
        for _, row in self.counts.iterrows():
            r = int(row["ring"])
            tag = "control" if r == self.n_rings + 1 else f"ring {r}"
            lines.append(
                f"  {tag:<14s} : {int(row['n_units']):d} units, "
                f"{int(row['n_unit_periods']):d} unit-periods   [{row['label']}]")
        lines.append(
            f"  time-varying   : {self.n_switchers} unit(s) change ring"
            + ("" if self.is_static else " — `unit_frame()` is unavailable"))
        d = self.frame["distance"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if d.size:
            lines.append(
                f"  nearest-treated distance: min {d.min():.4g}, "
                f"median {np.median(d):.4g}, max {d.max():.4g} {u}")
        if self.n_islands:
            lines.append(
                f"  islands        : {self.n_islands} unit(s) unreachable from "
                "any treated unit (assigned to the control ring)")
        return "\n".join(lines)

    def to_markdown(self, which: str = "counts", **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_frame`."""
        return _render(self.to_frame(which), "markdown", **kwargs)

    def to_latex(self, which: str = "counts", **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_frame`."""
        return _render(self.to_frame(which), "latex", **kwargs)

    def to_typst(self, which: str = "counts", **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_frame`."""
        return _render(self.to_frame(which), "typst", **kwargs)

    def plot(self, *, ax=None, figsize: tuple[float, float] = (9.0, 3.6)):
        """Diagnostic plot: distance histogram with the cut points, and a
        units-per-ring bar chart.

        Cut points should sit in a *thin* part of the distance distribution; a
        cut through a mode makes the ring boundary arbitrary. This is a
        diagnostic, not a map — choropleths need a geometry stack and stay out
        of scope.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            When given, only the histogram panel is drawn onto it.
        figsize : tuple of float

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        u = self.distance_unit
        d = self.frame["distance"].to_numpy(dtype=float)
        d = d[np.isfinite(d)]
        if ax is not None:
            axes = [ax]
            fig = ax.figure
        else:
            fig, axarr = plt.subplots(1, 2, figsize=figsize)
            axes = list(np.atleast_1d(axarr))
        a0 = axes[0]
        if d.size:
            a0.hist(d, bins=min(40, max(5, int(np.sqrt(d.size)))),
                    color="steelblue", alpha=0.75)
        for e in self.edges[1:]:
            a0.axvline(float(e), color="firebrick", ls="--", lw=1.0)
        a0.set_xlabel(f"distance to nearest treated unit ({u})")
        a0.set_ylabel("unit-periods")
        a0.set_title("Distance distribution and band edges")
        if len(axes) > 1:
            a1 = axes[1]
            c = self.counts
            colors = ["darkorange" if int(r) == self.n_rings + 1 else "steelblue"
                      for r in c["ring"]]
            a1.bar(np.arange(len(c)), c["n_units"].to_numpy(float), color=colors)
            a1.set_xticks(np.arange(len(c)))
            a1.set_xticklabels(list(c["label"]), rotation=45, ha="right", fontsize=8)
            a1.set_ylabel("units ever in ring")
            a1.set_title("Units per ring")
            a1.annotate("omitted category", xy=(len(c) - 1, 0),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=8, color="darkorange")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Ring builders
# ---------------------------------------------------------------------------
def _build_assignment(
    *,
    ids: tuple,
    times: pd.Index,
    entry_pos: np.ndarray,
    D_static: np.ndarray | None,
    D_by_cohort: np.ndarray | None,
    A_static: np.ndarray | None,
    A_by_cohort: np.ndarray | None,
    cohort_pos: np.ndarray,
    edges: np.ndarray,
    labels: tuple,
    ring_timing: str,
    metric: str,
    distance_unit: str,
    n_islands: int,
) -> RingAssignment:
    """Assemble the long assignment frame (shared by both ring builders)."""
    n, T = len(ids), len(times)
    R = len(edges) - 1
    dist = np.empty((n, T))
    near = np.empty((n, T), dtype=int)
    if ring_timing == "already_treated":
        # k(t) = number of cohorts that have switched on by period t.
        kt = np.searchsorted(cohort_pos, np.arange(T), side="right") - 1
        for t in range(T):
            if kt[t] < 0:
                dist[:, t] = np.inf
                near[:, t] = -1
            else:
                dist[:, t] = D_by_cohort[kt[t]]
                near[:, t] = A_by_cohort[kt[t]]
    else:
        dist[:] = D_static[:, None]
        near[:] = A_static[:, None]

    code = _codes_from_distance(dist, edges)
    if ring_timing == "ever_treated":
        # Exposure starts at the nearest ever-treated unit's own adoption date
        # (a treated unit's own date for itself).
        a_pos = np.where(near[:, 0] >= 0, entry_pos[np.maximum(near[:, 0], 0)], T)
        a_pos = np.where(entry_pos < T, np.minimum(a_pos, entry_pos), a_pos)
        not_yet = np.arange(T)[None, :] < a_pos[:, None]
        code = np.where(not_yet, R + 1, code)
        dist = np.where(not_yet, np.inf, dist)
        near = np.where(not_yet, -1, near)
    # Ring 0 is absorbing: a treated unit stays in ring 0 for ever after.
    treated_now = np.arange(T)[None, :] >= entry_pos[:, None]
    code = np.where(treated_now, 0, code)
    dist = np.where(treated_now, 0.0, dist)

    has_entry, a_idx, entry_ring, ev, switch = _event_clock(code, R)
    n_switchers = int(switch.sum())

    entry_ring_col = pd.array(
        np.repeat(np.where(has_entry, entry_ring, 0), T), dtype="Int64")
    entry_ring_col[~np.repeat(has_entry, T)] = pd.NA

    ids_arr = np.empty(len(ids), dtype=object)
    ids_arr[:] = list(ids)
    times_arr = np.empty(len(times), dtype=object)
    times_arr[:] = list(times)
    unit_col = np.repeat(ids_arr, T)
    time_col = np.tile(times_arr, n)
    entry_time = np.array(
        [times[a_idx[i]] if has_entry[i] else None for i in range(n)], dtype=object)
    frame = pd.DataFrame({
        "unit": unit_col,
        "time": time_col,
        "distance": dist.ravel(),
        "ring": code.ravel().astype(np.int64),
        "ring_label": [labels[c] for c in code.ravel()],
        "entry_ring": entry_ring_col,
        "entry_time": np.repeat(entry_time, T),
        "event_time": ev.ravel(),
        "nearest_treated": [ids[j] if j >= 0 else None for j in near.ravel()],
        "treated": treated_now.ravel(),
    })

    rows = []
    for r in range(R + 2):
        mask = code == r
        rows.append({
            "ring": r,
            "label": labels[r],
            "lo_edge": (np.nan if r == 0 else
                        (float(edges[r - 1]) if r <= R else float(edges[-1]))),
            "hi_edge": (np.nan if r == 0 else
                        (float(edges[r]) if r <= R else np.inf)),
            "n_units": int(mask.any(axis=1).sum()),
            "n_unit_periods": int(mask.sum()),
        })
    counts = pd.DataFrame(rows)

    n_treated = int((entry_pos < T).sum())
    n_control = int((~has_entry).sum())
    return RingAssignment(
        frame=frame,
        counts=counts,
        ids=tuple(ids),
        times=tuple(times),
        edges=tuple(float(e) for e in edges),
        labels=tuple(labels),
        n_rings=int(R),
        ring_timing=ring_timing,
        metric=metric,
        distance_unit=distance_unit,
        is_static=bool(n_switchers == 0),
        n_switchers=n_switchers,
        n_treated=n_treated,
        n_control=n_control,
        n_islands=int(n_islands),
    )


def _finish_rings(
    *,
    func: str,
    ids: tuple,
    D: np.ndarray,
    treated_idx: np.ndarray,
    entry_pos: np.ndarray,
    times: pd.Index,
    edges: np.ndarray,
    labels: tuple,
    ring_timing: str,
    metric: str,
    distance_unit: str,
    n_islands: int,
    warn_zero_distance: bool,
) -> RingAssignment:
    """Shared tail of :func:`exposure_rings` and :func:`contiguity_rings`."""
    n = len(ids)
    R = len(edges) - 1
    treated_entry = entry_pos[treated_idx]
    cohort_pos, treated_cohort = np.unique(treated_entry, return_inverse=True)
    n_cohorts = len(cohort_pos)

    if warn_zero_distance and D.size:
        untreated = np.setdiff1d(np.arange(n), treated_idx)
        if untreated.size and np.any(D[untreated] == 0.0):
            warnings.warn(
                f"{func}: an untreated unit sits at distance exactly 0 from a "
                "treated unit; that usually signals a bad crosswalk rather than "
                "genuine co-location (it is coded into ring 1)",
                RuntimeWarning, stacklevel=3)

    if ring_timing == "already_treated":
        M, A = _running_min_by_cohort(D, treated_cohort, treated_idx, n_cohorts)
        assign = _build_assignment(
            ids=ids, times=times, entry_pos=entry_pos,
            D_static=None, D_by_cohort=M, A_static=None, A_by_cohort=A,
            cohort_pos=cohort_pos, edges=edges, labels=labels,
            ring_timing=ring_timing, metric=metric,
            distance_unit=distance_unit, n_islands=n_islands)
    else:
        j = np.argmin(D, axis=1) if D.shape[1] else np.zeros(n, dtype=int)
        d = D[np.arange(n), j] if D.shape[1] else np.full(n, np.inf)
        a = treated_idx[j] if D.shape[1] else np.full(n, -1)
        a = np.where(np.isfinite(d), a, -1)
        assign = _build_assignment(
            ids=ids, times=times, entry_pos=entry_pos,
            D_static=d, D_by_cohort=None, A_static=a, A_by_cohort=None,
            cohort_pos=cohort_pos, edges=edges, labels=labels,
            ring_timing=ring_timing, metric=metric,
            distance_unit=distance_unit, n_islands=n_islands)

    if int(assign.counts.loc[assign.counts["ring"] == R + 1,
                             "n_unit_periods"].iloc[0]) == 0:
        warnings.warn(
            f"{func}: every unit-period lies inside the outermost ring "
            f"(rings[-1]={_fmt_edge(edges[-1])}); there is no clean control "
            "group, and `spatial_did` will refuse to fit. Widen the sample or "
            "shrink `rings`.",
            RuntimeWarning, stacklevel=3)
    return assign


def exposure_rings(
    coords: Any,
    *,
    treat_time: Any,
    times: Any | None = None,
    rings: Sequence[float] = (0.0, 25.0, 50.0, 100.0),
    ring_timing: str = "already_treated",
    metric: str = "haversine",
    ids: Sequence[Any] | None = None,
    labels: Sequence[str] | None = None,
) -> RingAssignment:
    """Build a ``(unit, time)`` exposure-ring assignment from coordinates.

    Distance to the nearest treated unit, the ring code, the entry ring and the
    event clock — usable on its own to inspect and defend the ring cut points
    before estimating anything.

    Parameters
    ----------
    coords : DataFrame, mapping or array
        A DataFrame indexed by unit id whose first two columns are the
        coordinates, a mapping ``id -> (lat, lon)``, or an ``(n, 2)`` array
        together with ``ids``. Latitude first for ``metric='haversine'``.
    treat_time : Series, mapping or array
        Per-unit first-treatment period. ``NaN`` / ``None`` / ``inf`` mean
        never-treated. Arrays must be in ``ids`` order.
    times : sequence, optional
        The period labels. **Required** when ``ring_timing='already_treated'``
        (the ring code is then a function of ``t``); when
        ``ring_timing='ever_treated'`` it is still used to lay out the long
        frame, and defaults to the sorted distinct treat times.
    rings : sequence of float, default (0.0, 25.0, 50.0, 100.0)
        Band cut points, strictly increasing and starting at ``0.0``. Band 1 is
        ``[0, rings[1]]``, band ``r >= 2`` is ``(rings[r-1], rings[r]]``.
    ring_timing : {'already_treated', 'ever_treated'}, default 'already_treated'
        Whether exposure is measured against the units treated *so far*
        (default) or against every ever-treated unit. See Notes for the exact
        definitions and the module docstring for why the default is the default.
    metric : {'haversine', 'euclidean'}, default 'haversine'
    ids : sequence, optional
        Unit labels when ``coords`` is a bare array.
    labels : sequence of str, optional
        Ring labels, length ``len(rings) + 1`` (ring 0, the ``R`` bands, and
        the control group).

    Returns
    -------
    RingAssignment

    Raises
    ------
    ValueError
        ``rings`` not strictly increasing / not starting at 0.0 / shorter than
        two / non-finite / negative; unknown ``ring_timing`` or ``metric``;
        ``times`` omitted with ``ring_timing='already_treated'``; ``labels`` of
        the wrong length; no unit has a finite treat time; duplicate unit ids.
    KeyError
        ``treat_time`` missing a unit that appears in ``coords`` (the message
        names up to three of them).

    Warns
    -----
    RuntimeWarning
        A treated and an untreated unit at distance exactly 0; every unit
        falling inside the outermost ring.

    Notes
    -----
    Ties among equidistant treated units cannot change the ring code — the
    distance is a minimum — so the assignment is always unique. Ties only
    affect the reported ``nearest_treated`` identity, where the earliest cohort
    wins and, within a cohort, the first unit in ``ids`` order.

    The two timings, precisely. Write ``g_j`` for unit ``j``'s first treated
    period and ``d_ij`` for the distance.

    * ``'already_treated'``: ``D_it = min{d_ij : g_j <= t}`` (``+inf`` before
      anything nearby is treated). The set of already-treated units only grows,
      so ``D_it`` is weakly decreasing and each unit walks *inward* through a
      decreasing sequence of ring codes. That monotonicity is what makes
      ``entry_time`` and ``entry_ring`` well defined, and it is why the result
      reports ``n_switchers``.
    * ``'ever_treated'``: ``D_i = min_j d_ij`` over every ever-treated ``j``,
      time-invariant, switched on at ``a_i = g_{j*}`` where ``j*`` is that
      nearest unit (a treated unit uses its own ``g_i``). Before ``a_i`` the
      unit sits in the control code. It gives one ring per unit — which is what
      an estimator with a scalar per-unit treat time needs — at the cost of
      ignoring exposure to a *nearer-in-time but farther-away* treated unit: in
      the two-cohort example above a unit 40 km from a t=1 plant and 10 km from
      a t=5 plant is coded ring 2 from t=1 under ``'already_treated'`` and stays
      in the control group until t=5 under ``'ever_treated'``.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.did.spatial_did import exposure_rings
    >>> xy = pd.DataFrame({'x': [0.0, 10.0, 40.0, 200.0], 'y': [0.0] * 4},
    ...                   index=['a', 'b', 'c', 'd'])
    >>> ra = exposure_rings(xy, treat_time={'a': 2, 'b': np.nan, 'c': np.nan,
    ...                                     'd': np.nan},
    ...                     times=[0, 1, 2, 3], metric='euclidean')
    >>> ra.frame.loc[ra.frame.unit == 'b', 'ring'].tolist()
    [4, 4, 1, 1]
    """
    if ring_timing not in _RING_TIMINGS:
        raise ValueError(
            f"exposure_rings: ring_timing must be one of {_RING_TIMINGS}, "
            f"got {ring_timing!r}")
    if metric not in _METRICS:
        raise ValueError(
            f"exposure_rings: metric must be one of {_METRICS}, got {metric!r}")
    edges = _validate_edges(rings, "exposure_rings")
    R = len(edges) - 1
    unit_name = "km" if metric == "haversine" else "units"
    if labels is None:
        lab = _default_labels(edges, unit_name)
    else:
        lab = tuple(str(s) for s in labels)
        if len(lab) != R + 2:
            raise ValueError(
                f"exposure_rings: labels must have {R + 2} entries (ring 0, "
                f"{R} band(s) and the control group), got {len(lab)}")

    arr, unit_ids = _coerce_coords(coords, ids)
    tt = _coerce_treat_time(treat_time, unit_ids, "exposure_rings")

    if times is None:
        if ring_timing == "already_treated":
            raise ValueError(
                "exposure_rings: `times` is required when "
                "ring_timing='already_treated' — the ring code is a function "
                "of the period; pass the panel's sorted period labels")
        obs = sorted({v for v in tt if not _is_never(v)})
        if not obs:
            raise ValueError(
                "exposure_rings: no unit has a finite treat_time, so there is "
                "nothing to build rings around")
        periods = pd.Index(obs, tupleize_cols=False)
    else:
        periods = pd.Index(sorted(set(list(times))), tupleize_cols=False)
        if len(periods) == 0:
            raise ValueError("exposure_rings: `times` is empty")

    entry_pos = _entry_positions(periods, tt, "exposure_rings")
    treated_idx = np.flatnonzero(entry_pos < len(periods))
    if treated_idx.size == 0:
        raise ValueError(
            "exposure_rings: no unit has a finite treat_time inside the period "
            "range, so there is nothing to build rings around")

    D = _distance_block(arr, arr[treated_idx], metric)
    return _finish_rings(
        func="exposure_rings", ids=unit_ids, D=D, treated_idx=treated_idx,
        entry_pos=entry_pos, times=periods, edges=edges, labels=lab,
        ring_timing=ring_timing, metric=metric, distance_unit=unit_name,
        n_islands=0, warn_zero_distance=True)


def contiguity_rings(
    W: SpatialWeights,
    *,
    treat_time: Any,
    times: Any | None = None,
    orders: int = 3,
    ring_timing: str = "already_treated",
    labels: Sequence[str] | None = None,
) -> RingAssignment:
    """Exposure rings measured in border crossings rather than kilometres.

    Ring ``r`` is "``r`` hops from the nearest treated unit" on the adjacency
    graph — the Berg-Reisinger-Streitz / Delgado-Florax contiguity-order
    version of the design. Unweighted shortest paths are computed from the
    treated units only, which costs ``O(N_treated (E + N log N))`` on a sparse
    graph and is far cheaper than the dense distance route.

    Parameters
    ----------
    W : SpatialWeights
        Adjacency; only its sparsity pattern is used (``W.binary()``).
    treat_time : Series, mapping or array
        Per-unit first-treatment period, keyed by / ordered as ``W.ids``.
    times : sequence, optional
        Period labels; required when ``ring_timing='already_treated'``.
    orders : int, default 3
        Highest hop count that gets its own ring. Capped at 10: beyond a
        handful of hops the bands stop being interpretable and the outside
        group empties out.
    ring_timing : {'already_treated', 'ever_treated'}, default 'already_treated'
    labels : sequence of str, optional
        Length ``orders + 2``.

    Returns
    -------
    RingAssignment
        With ``metric='graph'`` and ``distance_unit='hops'``. Islands and units
        in a component containing no treated unit get ``+inf`` hops and land in
        the control ring; ``n_islands`` echoes ``W.n_islands``.

    Raises
    ------
    ValueError
        ``W`` is not a :class:`~puremacro.spatial.weights.SpatialWeights`;
        ``orders`` outside ``1..10``; unknown ``ring_timing``; ``times``
        omitted with ``ring_timing='already_treated'``; ``labels`` of the wrong
        length; no treated units.
    KeyError
        ``treat_time`` missing a unit in ``W.ids``.

    Warns
    -----
    RuntimeWarning
        ``W`` has islands; every unit lies within ``orders`` hops.

    Examples
    --------
    >>> import numpy as np
    >>> from puremacro.spatial.weights import SpatialWeights
    >>> import scipy.sparse as sp
    >>> A = sp.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0],
    ...                             [0, 1, 0, 1], [0, 0, 1, 0]], float))
    >>> W = SpatialWeights(A, ids=('a', 'b', 'c', 'd'))
    >>> ra = contiguity_rings(W, treat_time={'a': 1, 'b': None, 'c': None,
    ...                                      'd': None},
    ...                       times=[0, 1], orders=2)
    >>> ra.frame.loc[ra.frame.time == 1, 'ring'].tolist()
    [0, 1, 2, 3]
    """
    if not isinstance(W, SpatialWeights):
        raise ValueError(
            "contiguity_rings: W must be a puremacro.spatial.weights."
            f"SpatialWeights, got {type(W).__name__}")
    if ring_timing not in _RING_TIMINGS:
        raise ValueError(
            f"contiguity_rings: ring_timing must be one of {_RING_TIMINGS}, "
            f"got {ring_timing!r}")
    orders = int(orders)
    if orders < 1 or orders > 10:
        raise ValueError(
            f"contiguity_rings: orders must lie in 1..10, got {orders}")
    edges = np.arange(orders + 1, dtype=float)
    if labels is None:
        lab = _default_labels(edges, "hops")
    else:
        lab = tuple(str(s) for s in labels)
        if len(lab) != orders + 2:
            raise ValueError(
                f"contiguity_rings: labels must have {orders + 2} entries, "
                f"got {len(lab)}")

    unit_ids = tuple(W.ids)
    tt = _coerce_treat_time(treat_time, unit_ids, "contiguity_rings")
    if times is None:
        if ring_timing == "already_treated":
            raise ValueError(
                "contiguity_rings: `times` is required when "
                "ring_timing='already_treated'")
        obs = sorted({v for v in tt if not _is_never(v)})
        if not obs:
            raise ValueError("contiguity_rings: no unit has a finite treat_time")
        periods = pd.Index(obs, tupleize_cols=False)
    else:
        periods = pd.Index(sorted(set(list(times))), tupleize_cols=False)
    entry_pos = _entry_positions(periods, tt, "contiguity_rings")
    treated_idx = np.flatnonzero(entry_pos < len(periods))
    if treated_idx.size == 0:
        raise ValueError(
            "contiguity_rings: no unit has a finite treat_time inside the "
            "period range")
    if W.n_islands:
        warnings.warn(
            f"contiguity_rings: W has {W.n_islands} island(s) with no "
            "neighbour; they are unreachable and land in the control ring",
            RuntimeWarning, stacklevel=2)

    B = W.binary().W
    hops = shortest_path(B, method="D", unweighted=True, directed=False,
                         indices=treated_idx)
    D = np.atleast_2d(hops).T if treated_idx.size == 1 else np.asarray(hops).T
    D = D.reshape(W.n, treated_idx.size)
    n_islands = int(np.sum(~np.isfinite(D).any(axis=1)))
    return _finish_rings(
        func="contiguity_rings", ids=unit_ids, D=D, treated_idx=treated_idx,
        entry_pos=entry_pos, times=periods, edges=edges, labels=lab,
        ring_timing=ring_timing, metric="graph", distance_unit="hops",
        n_islands=n_islands, warn_zero_distance=False)


# ---------------------------------------------------------------------------
# Covariance machinery
# ---------------------------------------------------------------------------
def _dk_bandwidth(n_periods: int) -> int:
    """Driscoll-Kraay bandwidth ``max(1, floor(4 (T/100)^(2/9)))`` on periods.

    Identical to ``lp._panel_helpers.make_conley_se_fn`` / ``_focal_dk_se``, so
    a spatial DiD and a panel LP on the same panel use the same time kernel.
    """
    return max(1, int(np.floor(4 * (n_periods / 100) ** (2 / 9))))


def _cluster_meat(X: np.ndarray, u: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """``sum_g s_g s_g'`` with ``s_g`` the within-cluster score sum."""
    score = X * u[:, None]
    codes, inv = np.unique(np.asarray(keys), return_inverse=True)
    S_g = np.zeros((len(codes), X.shape[1]))
    np.add.at(S_g, inv, score)
    return S_g.T @ S_g


def _psd_clip(S: np.ndarray) -> np.ndarray:
    """Nearest PSD matrix in the Frobenius sense: clip eigenvalues at zero."""
    A = 0.5 * (S + S.T)
    w, V = np.linalg.eigh(A)
    return (V * np.clip(w, 0.0, None)) @ V.T


def _sandwich(
    fit: dict,
    *,
    cov_type: str,
    coords_df: pd.DataFrame | None,
    cutoff_km: float | None,
    time_lags: int,
    kernel: str,
    metric: str,
    cluster_keys: np.ndarray | None,
    small_sample: bool,
    psd_adjust: str,
    func: str,
    quiet: bool = False,
) -> dict:
    """Covariance of a within fit under ``cov_type``, with the PSD bookkeeping.

    Returns a dict with ``vcov``, ``se``, ``meat_min_eig``, ``dof_factor``,
    ``k_total`` and ``n_negative_variance``. A non-positive sandwich variance
    (the two-dimensional Bartlett kernel is not positive definite, so the meat
    can be indefinite) yields ``NaN`` for that coefficient's SE rather than a
    silent ``sqrt`` of a negative number or a zero standard error, matching
    :func:`puremacro.spatial.spatial_panel`.
    """
    X = fit["X_within"]
    u = fit["residuals"]
    XtX_inv = fit["XtX_inv"]
    ent = np.asarray(fit["entity_keys"])
    tim = np.asarray(fit["time_keys"])
    n_obs, k = X.shape
    n_units = int(len(np.unique(ent)))
    n_periods = int(len(np.unique(tim)))
    k_total = int(k + n_units + n_periods - 1)

    if cov_type == "conley":
        from ..spatial.hac import spatial_hac_panel_meat

        S = spatial_hac_panel_meat(
            X, u, coords_df, ent, tim, float(cutoff_km), int(time_lags),
            kernel=kernel, metric=metric)
        dof = 1.0
        if small_sample:
            if n_obs > k_total:
                dof = n_obs / (n_obs - k_total)
            elif not quiet:
                warnings.warn(
                    f"{func}: the absorbed fixed effects use up every degree of "
                    f"freedom ({k_total} parameters for {n_obs} observations); "
                    "the small-sample correction is set to 1",
                    RuntimeWarning, stacklevel=3)
    else:
        keys = ent if cluster_keys is None else np.asarray(cluster_keys)
        S = _cluster_meat(X, u, keys)
        G = int(len(np.unique(keys)))
        if small_sample and G > 1 and n_obs > k_total:
            dof = (G / (G - 1.0)) * ((n_obs - 1.0) / (n_obs - k_total))
        else:
            dof = 1.0
        if G < 20 and not quiet:
            warnings.warn(
                f"{func}: only {G} clusters; the normal critical values are "
                "optimistic and the cluster-robust variance is badly biased",
                RuntimeWarning, stacklevel=3)

    S_sym = 0.5 * (S + S.T)
    min_eig = float(np.linalg.eigvalsh(S_sym).min()) if k else float("nan")
    if min_eig < 0:
        if psd_adjust == "warn" and not quiet:
            warnings.warn(
                f"{func}: the {kernel} HAC meat is indefinite (smallest "
                f"eigenvalue {min_eig:.3g}); the two-dimensional Bartlett "
                "kernel is not positive definite. Joint Wald statistics use a "
                "pseudo-inverse; pass psd_adjust='clip' for a conservative PSD "
                "projection.",
                RuntimeWarning, stacklevel=3)
        elif psd_adjust == "clip":
            S = _psd_clip(S)
    V = XtX_inv @ S @ XtX_inv * dof
    V = 0.5 * (V + V.T)
    d = np.diag(V)
    n_neg = int(np.sum(d <= 0))
    if n_neg and not quiet:
        warnings.warn(
            f"{func}: {n_neg} sandwich variance(s) came out non-positive on an "
            "indefinite meat; their se / t / p / CI are NaN. psd_adjust='clip' "
            "removes them at the cost of larger (conservative) standard errors.",
            RuntimeWarning, stacklevel=3)
    se = np.where(d > 0, np.sqrt(np.where(d > 0, d, 1.0)), np.nan)
    return {
        "vcov": V,
        "se": se,
        "meat_min_eig": min_eig,
        "dof_factor": float(dof),
        "k_total": k_total,
        "n_negative_variance": n_neg,
    }


def _inference_row(b: float, se: float, z: float) -> dict:
    """``effect / se / t / p / lo / hi`` with NaN propagation."""
    if not np.isfinite(se) or se <= 0:
        return {"effect": b, "se": np.nan, "t": np.nan, "p": np.nan,
                "lo": np.nan, "hi": np.nan}
    t = b / se
    return {"effect": b, "se": se, "t": t, "p": float(2.0 * norm.sf(abs(t))),
            "lo": b - z * se, "hi": b + z * se}


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------
@dataclass(frozen=True, eq=False)
class SpatialDiDResult:
    """Result of :func:`spatial_did`.

    Every DataFrame is a private copy, so mutating the caller's frame — or a
    frame this object hands out — cannot change what the result reports.

    Attributes
    ----------
    ring_table : pd.DataFrame
        The headline table: one row per ring ``0..R`` plus a final row for the
        omitted control group. Columns ``ring``, ``label``, ``lo_edge``,
        ``hi_edge`` (in ``distance_unit``; NaN/inf on the treated and control
        rows), ``n_units`` (units ever in the ring), ``n_obs`` (unit-periods
        used), ``effect``, ``se``, ``t``, ``p``, ``lo``, ``hi`` (NaN on the
        control row and on dropped rings) and ``dropped`` (bool). It comes from
        the **static two-way-FE fit**, always.
    coef : np.ndarray
        ``(k,)`` static coefficients ``delta_0..delta_R`` over the kept rings.
    vcov : np.ndarray
        ``(k, k)`` covariance of ``coef`` under ``cov_type``, small-sample
        factor already applied.
    names : tuple of str
        Coefficient names aligned with ``coef``.
    ring_codes : tuple of int
        The ring codes actually estimated, aligned with ``coef``.
    naive_att : float
        The two-way-FE DiD that pools every untreated unit into the control
        group, computed **inside the same within projection and on the same
        estimation sample** — that is what makes the Frisch-Waugh identity
        exact. When ``n_obs_dropped > 0`` it is therefore not the number a
        practitioner would get from a TWFE regression on the untrimmed panel.
    naive_se : float
        Its standard error, from its own within residuals under the same
        ``cov_type``. A separate object from ``direct_se``.
    direct_effect, direct_se : float
        ``delta_0`` and its SE. Ring 0 is absorbing, so this is the *total*
        effect on the treated (own effect plus spillovers received from other
        treated units), which equals the direct ATT only when no two treated
        units lie within ``rings[-1]`` of each other.
    contamination : float
        ``naive_att - direct_effect = sum_r theta_r delta_r``, exact to machine
        precision.
    contamination_se : float
        ``sqrt(c'Vc)`` with ``c = (0, theta_1, ..., theta_R)``. ``theta`` is
        data-dependent but held fixed given ``X``, the same conditional-on-X
        convention the rest of the package uses, so this understates the
        unconditional uncertainty.
    theta : np.ndarray
        ``(len(coef) - 1,)`` Frisch-Waugh auxiliary coefficients, aligned with
        ``names[1:]`` / ``ring_codes[1:]``.
    total_effect, total_effect_se : float
        ``delta_0 + sum_{r>=1} (N_r / N_0) delta_r`` and ``sqrt(a'Va)``. The
        aggregate effect per treated unit, including its spillover footprint.
        ``N_0 = n_treated_units`` is the number of **ever-treated** units and
        ``N_r`` (``r >= 1``) the number of **never-treated** units whose entry
        ring is ``r``. That is a partition of the exposed units — a switcher
        is not double-counted, and neither is a late-cohort treated unit that
        spent its pre-period in some other unit's ring 2 (counting it as a
        spillover recipient while leaving it out of the denominator inflated
        every ratio). The two definitions coincide whenever no treated unit
        lies within ``rings[-1]`` of an earlier cohort.
    event_study : pd.DataFrame or None
        Columns ``ring``, ``label``, ``event_time``, ``att``, ``se``, ``t``,
        ``p``, ``lo``, ``hi``, ``n_obs``, ``n_cohorts``. ``None`` when
        ``event_study=False``. Cells are indexed by the **current** ring with
        the clock anchored at entry (before entry, by the entry ring), so a
        unit that migrates inward does not drag its old band's dynamic path
        with it.
    att_by_ring_es : pd.DataFrame or None
        The event study aggregated back to one number per ring over ``e >= 0``
        with post-cell-count weights: ``ring``, ``label``, ``att``, ``se``,
        ``t``, ``p``, ``lo``, ``hi``, ``n_post_cells``. Not equal to the static
        ``effect`` in general — the static coefficient is a GLS-type weighted
        average of the dynamic path, not a count-weighted one.
    pretrend : pd.DataFrame or None
        Per-ring Wald that every ``delta_{r,e}`` with ``e <= -2`` is zero.
        There is **no** all-rings joint row: it was measurably over-sized in
        every design small enough to check, and G6 says such a statistic does
        not ship (see the module docstring). Columns ``ring``,
        ``label``, ``n_pre``, ``stat``, ``df``, ``p``, ``underpowered``. A row
        with ``underpowered=True`` carries ``stat = p = NaN``: its restriction
        leans on a ring held by fewer than ``_MIN_UNITS_FOR_JOINT`` units,
        where the chi-square reference is measurably wrong (see the module
        docstring). ``df`` is still reported, so you can see what was refused.
    placebo : pd.DataFrame or None
        Same columns as ``ring_table``, and with the same meanings — in
        particular ``n_units`` / ``n_obs`` count the units and unit-periods
        carrying that ring's fake dummy inside the placebo sample, not the
        placebo sample's overall size. Refitted on the pre-exposure subsample
        with the entry date shifted back by ``placebo_periods``. Rings are
        indexed by the **entry** ring (the placebo sample is entirely
        pre-exposure, so there is no current ring to use). ``None`` when
        ``placebo_periods = 0`` or the placebo sample is unusable.
    tests : pd.DataFrame
        Columns ``test``, ``null``, ``stat``, ``df``, ``p``, ``underpowered``.
        ``outer_ring`` is a single-coefficient Wald and always reported; every
        other row is joint and is blanked (``stat = p = NaN``,
        ``underpowered=True``) when its restriction touches a ring carried by
        fewer than ``_MIN_UNITS_FOR_JOINT`` units.
    outer_ring_contrast_bound : float
        ``|delta_R| + z * se_R`` on the outermost kept spillover ring. It bounds
        the **difference** between that ring and the beyond-ring controls, not
        the level of the spillover there: a spillover common to both cancels
        out of ``delta_R`` exactly. NaN when there is no spillover ring.
    outer_ring_contrast_share : float
        The same bound as a fraction of ``|direct_effect|``; NaN when the
        direct effect is below ``1e-8`` in absolute value.
    ring_edges : tuple of float
        The band cut points as given.
    ring_labels : tuple of str
        Length ``R + 2``, including the control label.
    distance_unit : str
        ``'km'``, ``'units'``, ``'hops'`` — the unit of ``lo_edge`` /
        ``hi_edge`` and of ``cutoff_km`` — or ``'unknown'`` on the ``ring_col``
        route, where the codes came from the caller and no distance was ever
        computed.
    ring_timing, cov_type, kernel, metric, psd_adjust : str
        ``ring_timing`` is ``'unknown'`` on the ``ring_col`` route for the same
        reason: nothing there applied a timing rule.
    static_estimator : str
        Always ``'twfe'``. The headline never follows
        ``event_study_estimator``; see the module docstring.
    event_study_estimator : str
        The resolved event-study estimator (``'twfe'`` or ``'sunab'``), or
        ``'none'`` when ``event_study=False``.
    include_not_yet_treated_in_rings : bool
    min_units_per_ring : int
    small_sample : bool
    alpha : float
    cutoff_km : float or None
    time_lags : int or None
    n_obs : int
        Unit-periods used by the static fit.
    n_obs_dropped : int
        Rows of the input frame removed by the exclusion rules
        (``min_units_per_ring``, ``include_not_yet_treated_in_rings=False``,
        a dropped ring). A row with a non-finite outcome is never among them:
        such a frame is refused outright.
    n_units, n_periods, n_treated_units, n_control_units, n_cohorts : int
        ``n_control_units`` counts units never inside any ring at any period.
    n_ring_switchers : int
        Units whose ring changes after they first become exposed.
    n_treated_clusters : int or None
        Connected components of the treated-unit graph at radius ``cutoff_km``;
        ``None`` under ``cov_type='cluster'``. This is the number of spatially
        independent experiments, and it — not ``n_obs`` — governs how well the
        normal approximation works.
    meat_min_eig : float or None
        Smallest eigenvalue of the HAC meat; negative values flag the
        non-PSD two-dimensional Bartlett problem.
    n_negative_variance : int
        Coefficients whose sandwich variance came out non-positive (a zero
        variance is not a real standard error either, so it is blanked too;
        the name is kept for backward compatibility). Their SE, t, p
        and CI are NaN).
    dof_factor : float
        The small-sample multiplier actually applied.
    k_total : int
        ``k + n_units + n_periods - 1``, the parameter count behind
        ``dof_factor``.
    dropped_rings : tuple of int
        Ring codes excluded from the design matrix.
    assignment : RingAssignment or None
        The ring assignment used, when it was built from coordinates or passed
        in; ``None`` on the ``ring_col`` route.
    """

    ring_table: pd.DataFrame
    coef: np.ndarray
    vcov: np.ndarray
    names: tuple
    ring_codes: tuple
    naive_att: float
    naive_se: float
    direct_effect: float
    direct_se: float
    contamination: float
    contamination_se: float
    theta: np.ndarray
    total_effect: float
    total_effect_se: float
    event_study: pd.DataFrame | None
    att_by_ring_es: pd.DataFrame | None
    pretrend: pd.DataFrame | None
    placebo: pd.DataFrame | None
    tests: pd.DataFrame
    outer_ring_contrast_bound: float
    outer_ring_contrast_share: float
    ring_edges: tuple
    ring_labels: tuple
    distance_unit: str
    ring_timing: str
    static_estimator: str
    event_study_estimator: str
    cov_type: str
    cutoff_km: float | None
    time_lags: int | None
    kernel: str
    metric: str
    psd_adjust: str
    include_not_yet_treated_in_rings: bool
    min_units_per_ring: int
    small_sample: bool
    alpha: float
    n_obs: int
    n_obs_dropped: int
    n_units: int
    n_periods: int
    n_treated_units: int
    n_control_units: int
    n_cohorts: int
    n_ring_switchers: int
    n_treated_clusters: int | None
    meat_min_eig: float | None
    n_negative_variance: int
    dof_factor: float
    k_total: int
    dropped_rings: tuple
    assignment: RingAssignment | None

    # -- presentation -------------------------------------------------------
    _TABLES = ("rings", "event_study", "att_by_ring_es", "tests", "pretrend",
               "placebo")

    def _table(self, which: str) -> pd.DataFrame:
        mapping = {
            "rings": self.ring_table,
            "event_study": self.event_study,
            "att_by_ring_es": self.att_by_ring_es,
            "tests": self.tests,
            "pretrend": self.pretrend,
            "placebo": self.placebo,
        }
        if which not in mapping:
            raise ValueError(
                f"SpatialDiDResult: which must be one of {self._TABLES}, "
                f"got {which!r}")
        tab = mapping[which]
        if tab is None:
            available = tuple(k for k, v in mapping.items() if v is not None)
            raise ValueError(
                f"SpatialDiDResult: the {which!r} table was not computed; "
                f"available tables are {available}")
        return tab.copy()

    def to_frame(self, which: str = "rings") -> pd.DataFrame:
        """One of the result tables; ``'rings'`` (the headline) by default."""
        return self._table(which)

    def to_markdown(self, which: str = "rings", **kwargs: Any) -> str:
        """Markdown rendering of :meth:`to_frame`."""
        return _render(self._table(which), "markdown", **kwargs)

    def to_latex(self, which: str = "rings", **kwargs: Any) -> str:
        """LaTeX rendering of :meth:`to_frame`."""
        return _render(self._table(which), "latex", **kwargs)

    def to_typst(self, which: str = "rings", **kwargs: Any) -> str:
        """Typst rendering of :meth:`to_frame`."""
        return _render(self._table(which), "typst", **kwargs)

    def summary(self) -> str:
        """Full text report, including every caveat the design cannot avoid."""
        u = self.distance_unit
        pct = int(round(100 * (1 - self.alpha)))
        lines = [
            f"Spatial DiD with exposure rings ({self.ring_timing}, {self.metric})",
            f"  panel      : {self.n_units} units x {self.n_periods} periods = "
            f"{self.n_obs} obs used"
            + (f" ({self.n_obs_dropped} dropped)" if self.n_obs_dropped else "")
            + f"; {self.n_treated_units} treated, {self.n_control_units} clean "
            f"controls, {self.n_cohorts} cohort"
            + ("s" if self.n_cohorts != 1 else ""),
            "  rings      : "
            + " / ".join(_fmt_edge(e) for e in self.ring_edges) + f" {u}"
            + f"   headline: static twfe   event study: {self.event_study_estimator}",
            "  inference  : " + (
                f"Conley HAC, cutoff {_fmt_edge(self.cutoff_km)} {u}, "
                f"{self.time_lags} time lag"
                + ("s" if (self.time_lags or 0) != 1 else "")
                if self.cov_type == "conley"
                else "cluster-robust by unit")
            + f"; normal critical values, dof factor {self.dof_factor:.4f}",
            "",
            f"  ring   band              units    effect      se      "
            f"{pct}% CI                 p",
        ]
        for _, r in self.ring_table.iterrows():
            code = int(r["ring"])
            tag = "-" if code == len(self.ring_labels) - 1 else str(code)
            if not np.isfinite(r["effect"]):
                lines.append(
                    f"  {tag:<5s}  {str(r['label']):<16s} {int(r['n_units']):>6d}"
                    + ("         .       .   (omitted category)"
                       if code == len(self.ring_labels) - 1
                       else "         .       .   (dropped from the design)"))
                continue
            se = r["se"]
            if np.isfinite(se):
                lines.append(
                    f"  {tag:<5s}  {str(r['label']):<16s} {int(r['n_units']):>6d}"
                    f" {r['effect']:>+9.4f} {se:>7.4f}   "
                    f"[{r['lo']:+.4f}, {r['hi']:+.4f}]  {r['p']:>7.4f}")
            else:
                lines.append(
                    f"  {tag:<5s}  {str(r['label']):<16s} {int(r['n_units']):>6d}"
                    f" {r['effect']:>+9.4f}     nan   "
                    "(non-positive sandwich variance)")
        theta_txt = ", ".join(f"{t:+.3f}" for t in np.atleast_1d(self.theta))
        lines += [
            "",
            f"  naive DiD (no rings) : {self.naive_att:+.4f} ({self.naive_se:.4f})",
            f"  ring-adjusted direct : {self.direct_effect:+.4f} "
            f"({self.direct_se:.4f})",
            f"  contamination        : {self.contamination:+.4f} "
            f"({self.contamination_se:.4f})  = sum_r theta_r delta_r"
            + (f", theta = ({theta_txt})" if theta_txt else ""),
            f"  total effect / treated unit : {self.total_effect:+.4f} "
            f"({self.total_effect_se:.4f})",
            "",
        ]
        for _, r in self.tests.iterrows():
            df = r["df"]
            df_txt = "nan" if not np.isfinite(df) else str(int(df))
            if bool(r.get("underpowered", False)):
                lines.append(
                    f"  {str(r['test']):<28s} {str(r['null']):<26s} "
                    f"chi2({df_txt}) NOT REPORTED — a ring in this restriction "
                    f"is carried by < {_MIN_UNITS_FOR_JOINT} units")
                continue
            lines.append(
                f"  {str(r['test']):<28s} {str(r['null']):<26s} "
                f"chi2({df_txt}) = {r['stat']:.4g}, p = {r['p']:.4g}")
        if np.isfinite(self.outer_ring_contrast_bound):
            share = self.outer_ring_contrast_share
            share_txt = ("" if not np.isfinite(share)
                         else f" ({100 * share:.0f}% of the direct effect)")
            lines += [
                "",
                f"  outer-ring CONTRAST bound: {self.outer_ring_contrast_bound:.4f}"
                + share_txt,
                "    differences in spillover between the outermost ring "
                f"({self.ring_labels[-2]}) and the",
                f"    {self.ring_labels[-1]} controls larger than this are ruled "
                f"out at the {int(round(100 * self.alpha))}% level.",
                "    A spillover COMMON to both is not identified by this "
                "design — widen rings[-1] to probe it.",
            ]
        lines += ["", "  notes:"]
        n_blank = int(self.tests["underpowered"].sum()) if len(self.tests) else 0
        if self.pretrend is not None:
            n_blank += int(self.pretrend["underpowered"].sum())
        if n_blank:
            lines.append(
                f"    {n_blank} joint Wald row(s) are NOT REPORTED "
                f"(underpowered=True): a ring in the restriction is")
            lines.append(
                f"    carried by fewer than {_MIN_UNITS_FOR_JOINT} units, "
                "where the chi2 reference rejects ~0.5 of the")
            lines.append(
                "    time at a nominal 0.10. The per-coefficient rows above "
                "are unaffected.")
        lines.append(
            "    the ring table, the decomposition and every Wald test come "
            "from the STATIC two-way-FE fit")
        lines.append(
            "    (static_estimator='twfe'); event_study_estimator only governs "
            "the event study.")
        if self.n_cohorts > 1:
            lines.append(
                "    with more than one cohort that static fit carries the "
                "Goodman-Bacon (2021) /")
            lines.append(
                "    de Chaisemartin-D'Haultfoeuille (2020) negative-weight "
                "problem, once per ring;")
            lines.append(
                "    read `att_by_ring_es` beside it."
                if self.att_by_ring_es is not None else
                "    set event_study=True to see `att_by_ring_es` beside it.")
        if self.n_ring_switchers:
            lines.append(
                f"    {self.n_ring_switchers} unit(s) change ring after entry; "
                "the event study is defined on the")
            lines.append(
                "    CURRENT ring with the clock anchored at entry.")
        if self.cov_type == "cluster":
            lines.append(
                "    cluster-by-unit discards every cross-unit score product, "
                "which is exactly the")
            lines.append(
                "    correlation the ring design induces; it under-covers here. "
                "Prefer cov_type='conley'.")
        else:
            lines.append(
                f"    {self.n_treated_clusters} spatially separated treated "
                f"cluster(s) at the {_fmt_edge(self.cutoff_km)} {u} cutoff"
                + ("" if (self.n_treated_clusters or 0) >= 10 else
                   " — the normal critical"))
            if (self.n_treated_clusters or 0) < 10:
                lines.append(
                    "    values are optimistic; read the p-values as indicative.")
            lines.append(
                "    Conley SEs assume increasing-domain asymptotics; they say "
                "nothing under infill.")
            if self.meat_min_eig is not None and self.meat_min_eig < 0:
                lines.append(
                    f"    HAC meat is indefinite (min eigenvalue "
                    f"{self.meat_min_eig:.3g}); joint tests use a pseudo-inverse"
                    f", psd_adjust={self.psd_adjust!r}.")
        if self.n_negative_variance:
            lines.append(
                f"    {self.n_negative_variance} coefficient(s) have a "
                "non-positive sandwich variance; their SEs are NaN.")
        if self.dropped_rings:
            lines.append(
                f"    dropped rings: {list(self.dropped_rings)} (empty, or below "
                f"min_units_per_ring={self.min_units_per_ring}).")
        return "\n".join(lines)

    def plot(self, *, ax=None, figsize: tuple[float, float] = (10.0, 4.2),
             show_event_study: bool | None = None, title: str | None = None):
        """Ring effects (left) and, when available, the event study (right).

        The x-axis of the left panel is the ring *index*, not a distance: the
        outer band is unbounded, so a distance midpoint would lie about it. The
        dashed horizontal line is ``naive_att``, so the correction the ring
        design buys is visible at a glance; dropped rings appear as open
        markers at zero height.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Used for the left panel only; the event study is then suppressed.
        figsize : tuple of float
        show_event_study : bool, optional
            Force the second panel on or off. Default: on when
            ``event_study`` is not None and ``ax`` is None.
        title : str, optional

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        want_es = (self.event_study is not None if show_event_study is None
                   else bool(show_event_study))
        if show_event_study and self.event_study is None:
            raise ValueError(
                "SpatialDiDResult.plot: show_event_study=True but the result "
                "carries no event study (it was fitted with event_study=False)")
        if ax is not None:
            want_es = False
            fig, a0, a1 = ax.figure, ax, None
        elif want_es:
            fig, (a0, a1) = plt.subplots(1, 2, figsize=figsize)
        else:
            fig, a0 = plt.subplots(figsize=(figsize[0] / 2, figsize[1]))
            a1 = None

        tab = self.ring_table
        est = tab[tab["ring"] < len(self.ring_labels) - 1]
        xs = np.arange(len(est))
        eff = est["effect"].to_numpy(float)
        se = est["se"].to_numpy(float)
        z = norm.ppf(1 - self.alpha / 2)
        ok = np.isfinite(eff)
        colors = ["firebrick" if int(r) == 0 else "steelblue" for r in est["ring"]]
        a0.axhline(0.0, color="black", lw=1.0)
        a0.axhline(self.naive_att, color="darkgreen", ls="--", lw=1.2)
        a0.annotate("naive DiD", xy=(xs[-1], self.naive_att), xytext=(4, 3),
                    textcoords="offset points", color="darkgreen", fontsize=8)
        spill = xs[[int(r) >= 1 for r in est["ring"]]]
        if spill.size:
            a0.axvspan(spill.min() - 0.5, spill.max() + 0.5, color="grey",
                       alpha=0.08)
        for i in xs:
            if ok[i]:
                a0.plot([i, i], [eff[i] - z * se[i], eff[i] + z * se[i]],
                        color=colors[i], lw=1.4)
                a0.plot([i], [eff[i]], "o", color=colors[i], ms=6)
            else:
                a0.plot([i], [0.0], "o", mfc="none", color="grey", ms=7)
        a0.set_xticks(xs)
        a0.set_xticklabels(
            [lab if np.isfinite(e) else "n/a"
             for lab, e in zip(est["label"], eff)],
            rotation=30, ha="right", fontsize=8)
        a0.set_ylabel("effect")
        a0.set_title(title or "Effect by exposure ring")

        if a1 is not None:
            es = self.event_study
            a1.axhline(0.0, color="black", ls=":", lw=1.0)
            a1.axvline(-0.5, color="grey", ls="--", lw=1.0)
            for r, grp in es.groupby("ring", sort=True):
                g = grp.sort_values("event_time")
                a1.plot(g["event_time"], g["att"], marker="o", ms=3,
                        label=str(g["label"].iloc[0]))
                a1.fill_between(g["event_time"], g["lo"], g["hi"], alpha=0.15)
            a1.set_xlabel("event time (periods since entry)")
            a1.set_ylabel("effect")
            a1.set_title("Event study by ring")
            a1.legend(fontsize=7)
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Estimation plumbing
# ---------------------------------------------------------------------------
def _long_frame(df: pd.DataFrame, unit: str, time: str, needed: Sequence[str],
                func: str) -> pd.DataFrame:
    """Long-format view of ``df`` with ``unit`` / ``time`` as ordinary columns."""
    data = df
    if not (unit in data.columns and time in data.columns):
        data = data.reset_index()
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(
            f"{func}: column(s) {missing} are not in the frame; its columns are "
            f"{list(data.columns)[:12]} and its index levels are "
            f"{list(df.index.names)}")
    return data


def _codes_aligned(ra: RingAssignment, units: pd.Index, periods: pd.Index,
                   func: str) -> np.ndarray:
    """Ring codes from an assignment, reindexed onto ``units x periods``."""
    if tuple(ra.ids) == tuple(units) and tuple(ra.times) == tuple(periods):
        return ra.frame["ring"].to_numpy(dtype=np.int64).reshape(
            len(units), len(periods)).copy()
    piv = ra.frame.pivot(index="unit", columns="time", values="ring")
    miss_u = [u for u in units if u not in piv.index]
    miss_t = [t for t in periods if t not in piv.columns]
    if miss_u or miss_t:
        raise KeyError(
            f"{func}: the RingAssignment does not cover "
            + (f"{len(miss_u)} unit(s), e.g. {miss_u[:3]}" if miss_u else "")
            + (" and " if miss_u and miss_t else "")
            + (f"{len(miss_t)} period(s), e.g. {miss_t[:3]}" if miss_t else ""))
    out = piv.reindex(index=list(units), columns=list(periods)).to_numpy()
    if np.isnan(out.astype(float)).any():
        raise ValueError(
            f"{func}: the RingAssignment has holes over the panel's "
            "(unit, period) grid")
    return out.astype(np.int64)


def _build_design(code_row: np.ndarray, kept: Sequence[int], labels: Sequence[str],
                  existing: Sequence[str], func: str) -> tuple[dict, list]:
    """Ring dummy columns, with a collision check against the user's columns."""
    cols, names = {}, []
    for r in kept:
        nm = f"__ring{r}__"
        if nm in existing:
            raise ValueError(
                f"{func}: the generated design column {nm!r} collides with a "
                "column already in the frame; rename it")
        cols[nm] = (code_row == r).astype(float)
        names.append(nm)
    return cols, names


def _ring_name(r: int, label: str) -> str:
    return f"ring{r}_" + str(label).replace(" ", "")


def _event_study_fit(ctx: dict) -> dict:
    """The per-ring event study, its aggregation and the pre-trend tests."""
    func = "spatial_did"
    code = ctx["code"]
    R = ctx["R"]
    kept = ctx["kept"]
    labels = ctx["labels"]
    has_entry, a_idx, entry_ring = ctx["has_entry"], ctx["a_idx"], ctx["entry_ring"]
    iu, it = ctx["row_u"], ctx["row_t"]
    est = ctx["est"]
    z = ctx["z"]
    gated = ctx["gated"]

    n_rows = len(est)
    exposed_row = has_entry[iu]
    e_raw = np.where(exposed_row, it - a_idx[iu], np.nan)
    cur = code[iu, it]
    es_ring = np.where(exposed_row & (e_raw >= 0) & (cur >= 0) & (cur <= R),
                       cur, entry_ring[iu])

    window = ctx["event_window"]
    if window is None:
        e_bin = e_raw.copy()
    else:
        emin, emax = int(window[0]), int(window[1])
        e_bin = np.where(exposed_row, np.clip(e_raw, emin, emax), np.nan)

    # A cohort with no e = -1 cell has no reference period; drop it.
    keep_unit = np.ones(code.shape[0], dtype=bool)
    dropped_cohorts = []
    for c in np.unique(a_idx[has_entry]):
        rows = exposed_row & (a_idx[iu] == c)
        if not rows.any():
            continue
        if not np.any(e_bin[rows] == -1):
            dropped_cohorts.append(int(c))
            keep_unit[(a_idx == c) & has_entry] = False
    if dropped_cohorts:
        warnings.warn(
            f"{func}: cohort(s) entering at period position {dropped_cohorts} "
            "have no e = -1 reference cell; they are dropped from the event "
            "study and kept in the static model",
            RuntimeWarning, stacklevel=3)

    row_ok = keep_unit[iu]
    if not row_ok.any():
        raise ValueError(
            f"{func}: no cohort has an e = -1 reference period, so no event "
            "study is identified; pass event_study=False")

    active = row_ok & exposed_row & np.isfinite(e_bin) & (e_bin != -1)
    cells_re = sorted({(int(r), int(e))
                       for r, e in zip(es_ring[active], e_bin[active])
                       if int(r) in kept})
    if not cells_re:
        raise ValueError(
            f"{func}: the event study has no estimable cell; widen "
            "`event_window` or pass event_study=False")

    estimator = ctx["estimator"]
    if estimator == "sunab":
        cohort_row = np.where(exposed_row, a_idx[iu], -1)
        cells = sorted({(int(r), int(c), int(e))
                        for r, c, e in zip(es_ring[active], cohort_row[active],
                                           e_bin[active])
                        if int(r) in kept})
    else:
        cells = [(r, -1, e) for r, e in cells_re]

    if len(cells) > ctx["max_params"]:
        raise ValueError(
            f"{func}: the event study needs {len(cells)} dummies, above "
            f"max_params={ctx['max_params']}. Narrow `event_window`, pass "
            "event_study_estimator='twfe', or event_study=False")

    X = {}
    names = []
    for (r, c, e) in cells:
        m = active & (es_ring == r) & (e_bin == e)
        if c >= 0:
            m = m & (cohort_row == c)
        nm = (f"__es_r{r}_e{e}__" if c < 0 else f"__es_r{r}_c{c}_e{e}__")
        X[nm] = m.astype(float)
        names.append(nm)

    block = pd.DataFrame({nm: X[nm][row_ok] for nm in names},
                         index=est.index[row_ok])
    sub = pd.concat([est.loc[row_ok], block], axis=1)
    keep_cols = [nm for nm in names if sub[nm].to_numpy().any()]
    if not keep_cols:
        raise ValueError(f"{func}: every event-study dummy is empty")
    cells = [c for c, nm in zip(cells, names) if nm in keep_cols]

    panel, ent_lv, tim_lv = as_panel_index(
        sub, entity_level=ctx["unit"], time_level=ctx["time"],
        unit_col=ctx["unit"], time_col=ctx["time"], func=func)
    fit = two_way_fe_within(panel, y_col=ctx["outcome"], x_cols=keep_cols,
                            entity_level=ent_lv, time_level=tim_lv)
    cov = _sandwich(fit, func=func, quiet=True, **ctx["cov_kwargs"])
    beta, V = fit["beta"], cov["vcov"]

    # Aggregation to (ring, event time). For 'twfe' this is the identity; for
    # 'sunab' it is the linear cohort-share average, so the joint covariance
    # survives as A V A'.
    out_cells = sorted({(r, e) for (r, _c, e) in cells})
    counts = {}
    for j, (r, c, e) in enumerate(cells):
        m = active & (es_ring == r) & (e_bin == e)
        if c >= 0:
            m = m & (cohort_row == c)
        counts[j] = float(m.sum())
    A = np.zeros((len(out_cells), len(cells)))
    for i, (r, e) in enumerate(out_cells):
        idx = [j for j, (rr, _c, ee) in enumerate(cells) if rr == r and ee == e]
        tot = sum(counts[j] for j in idx)
        for j in idx:
            A[i, j] = counts[j] / tot if tot > 0 else 1.0 / len(idx)
    delta = A @ beta
    V_agg = A @ V @ A.T
    se = np.sqrt(np.where(np.diag(V_agg) >= 0, np.diag(V_agg), np.nan))

    rows = []
    for i, (r, e) in enumerate(out_cells):
        m = active & (es_ring == r) & (e_bin == e)
        row = {"ring": r, "label": labels[r], "event_time": e}
        row.update(_inference_row(float(delta[i]), float(se[i]), z))
        row["att"] = row.pop("effect")
        row["n_obs"] = int(m.sum())
        row["n_cohorts"] = int(len(np.unique(a_idx[iu][m])))
        rows.append(row)
    es_df = pd.DataFrame(rows)[
        ["ring", "label", "event_time", "att", "se", "t", "p", "lo", "hi",
         "n_obs", "n_cohorts"]]

    # Static-equivalent per ring: post-cell-count weights over e >= 0.
    att_rows, dyn_rows = [], []
    for r in kept:
        post = [i for i, (rr, e) in enumerate(out_cells) if rr == r and e >= 0]
        if not post:
            continue
        w = np.array([float(es_df["n_obs"].iloc[i]) for i in post])
        w = w / w.sum() if w.sum() > 0 else np.full(len(post), 1.0 / len(post))
        B = np.zeros(len(out_cells))
        B[post] = w
        att = float(B @ delta)
        v = float(B @ V_agg @ B)
        row = {"ring": r, "label": labels[r]}
        row.update(_inference_row(att, float(np.sqrt(v)) if v >= 0 else np.nan, z))
        row["att"] = row.pop("effect")
        row["n_post_cells"] = int(len(post))
        att_rows.append(row)
        if len(post) >= 2:
            Rm = np.zeros((len(post) - 1, len(out_cells)))
            for j in range(len(post) - 1):
                Rm[j, post[j]] = 1.0
                Rm[j, post[j + 1]] = -1.0
            stat, dfree, p = _wald(delta, V_agg, Rm, name=func)
            gate = gated([r])
            dyn_rows.append({
                "test": f"dynamics_ring{r}",
                "null": f"delta_{{{r},e}} constant for e >= 0",
                "stat": np.nan if gate else stat, "df": dfree,
                "p": np.nan if gate else p, "underpowered": gate})
    att_df = pd.DataFrame(att_rows)[
        ["ring", "label", "att", "se", "t", "p", "lo", "hi", "n_post_cells"]] \
        if att_rows else None

    # Pre-trends: every delta_{r,e} with e <= -2 is zero.
    pre_rows = []
    for r in kept:
        pre = [i for i, (rr, e) in enumerate(out_cells) if rr == r and e <= -2]
        if not pre:
            continue
        Rm = np.zeros((len(pre), len(out_cells)))
        for j, i in enumerate(pre):
            Rm[j, i] = 1.0
        stat, dfree, p = _wald(delta, V_agg, Rm, name=func)
        gate = gated([r])
        pre_rows.append({"ring": r, "label": labels[r], "n_pre": len(pre),
                         "stat": np.nan if gate else stat, "df": dfree,
                         "p": np.nan if gate else p, "underpowered": gate})
    # The ALL-RINGS joint pre-trend row does not ship (G6). Measured on seeded
    # nulls at a nominal 0.10 it rejected 0.515 (90 units / 9 treated), 0.20
    # and 0.18 (two seed blocks of 200 units / 22 treated, every ring clearing
    # the size gate), 0.135 (200 / 20), 0.095 (300 / 30) and 0.087 (400 / 45):
    # twelve restrictions against a robust covariance carried by a few dozen
    # treated units. The per-ring rows above answer the same question with
    # three restrictions each and are approximately sized. See the module
    # docstring, "Deliberately omitted".
    pre_df = pd.DataFrame(pre_rows) if pre_rows else None
    return {"event_study": es_df, "att_by_ring_es": att_df,
            "pretrend": pre_df, "dynamics": dyn_rows,
            "agg_matrix": A, "cells": cells, "beta": beta, "vcov": V,
            "delta": delta, "vcov_agg": V_agg, "out_cells": out_cells}


def _placebo_fit(ctx: dict, k: int) -> pd.DataFrame | None:
    """Refit the static specification on the pre-exposure subsample with the
    entry date shifted back by ``k`` periods."""
    func = "spatial_did"
    R, kept, labels = ctx["R"], ctx["kept"], ctx["labels"]
    has_entry, a_idx, entry_ring = ctx["has_entry"], ctx["a_idx"], ctx["entry_ring"]
    iu, it = ctx["row_u"], ctx["row_t"]
    est, z = ctx["est"], ctx["z"]

    exposed_row = has_entry[iu]
    e_raw = np.where(exposed_row, it - a_idx[iu], np.nan)
    ok_cohort = np.ones(len(has_entry), dtype=bool)
    dropped = []
    for c in np.unique(a_idx[has_entry]):
        if c < k + 1:
            dropped.append(int(c))
            ok_cohort[(a_idx == c) & has_entry] = False
    if dropped:
        warnings.warn(
            f"{func}: cohort(s) at period position {dropped} have fewer than "
            f"{k + 1} pre-periods and are dropped from the placebo",
            RuntimeWarning, stacklevel=3)

    rows = ok_cohort[iu] & ((~exposed_row) | (e_raw < 0))
    if rows.sum() < 4 or not (exposed_row & rows).any():
        warnings.warn(
            f"{func}: the placebo sample is empty or has no exposed unit; "
            "placebo is None",
            RuntimeWarning, stacklevel=3)
        return None

    fake_on = exposed_row & (it >= (a_idx[iu] - k))
    cols, names = {}, []
    ring_of = np.where(exposed_row, entry_ring[iu], R + 1)
    for r in kept:
        v = ((ring_of == r) & fake_on).astype(float)
        if v[rows].any() and not np.all(v[rows] == v[rows][0]):
            cols[f"__pl{r}__"] = v
            names.append((r, f"__pl{r}__"))
    if not names:
        warnings.warn(
            f"{func}: the placebo design matrix has no variation; "
            "placebo is None",
            RuntimeWarning, stacklevel=3)
        return None

    block = pd.DataFrame({nm: cols[nm][rows] for _r, nm in names},
                         index=est.index[rows])
    sub = pd.concat([est.loc[rows], block], axis=1)
    panel, ent_lv, tim_lv = as_panel_index(
        sub, entity_level=ctx["unit"], time_level=ctx["time"],
        unit_col=ctx["unit"], time_col=ctx["time"], func=func)
    try:
        fit = two_way_fe_within(panel, y_col=ctx["outcome"],
                                x_cols=[nm for _r, nm in names],
                                entity_level=ent_lv, time_level=tim_lv)
    except np.linalg.LinAlgError:
        warnings.warn(
            f"{func}: the placebo design is collinear after the two-way "
            "projection; placebo is None",
            RuntimeWarning, stacklevel=3)
        return None
    cov = _sandwich(fit, func=func, quiet=True, **ctx["cov_kwargs"])
    # `n_units` / `n_obs` mean the same thing here as in `ring_table`: the
    # units and unit-periods CARRYING this ring's (fake) dummy, not the
    # sample-wide totals. Reporting the sample size on every row under the
    # ring table's column names would silently give two quantities one name.
    fit_units = pd.Index(np.unique(np.asarray(fit["entity_keys"])))
    unit_ser = est.loc[rows, ctx["unit"]].reset_index(drop=True)
    in_fit = unit_ser.isin(fit_units).to_numpy()
    out = []
    for j, (r, nm) in enumerate(names):
        carried = (cols[nm][rows] > 0) & in_fit
        row = {"ring": r, "label": labels[r],
               "lo_edge": ctx["lo_edge"][r], "hi_edge": ctx["hi_edge"][r],
               "n_units": int(unit_ser[carried].nunique()),
               "n_obs": int(carried.sum())}
        row.update(_inference_row(float(fit["beta"][j]), float(cov["se"][j]), z))
        row["dropped"] = False
        out.append(row)
    return pd.DataFrame(out)[
        ["ring", "label", "lo_edge", "hi_edge", "n_units", "n_obs", "effect",
         "se", "t", "p", "lo", "hi", "dropped"]]


def spatial_did(
    df: pd.DataFrame,
    *,
    unit: str = "unit",
    time: str = "time",
    outcome: str = "y",
    treat_time: str = "treat_time",
    coords: Any | None = None,
    assignment: RingAssignment | None = None,
    ring_col: str | None = None,
    rings: Sequence[float] = (0.0, 25.0, 50.0, 100.0),
    ring_timing: Any = _UNSET,
    metric: Any = _UNSET,
    include_not_yet_treated_in_rings: bool = True,
    min_units_per_ring: int = 1,
    event_study: bool = True,
    event_study_estimator: str = "auto",
    event_window: tuple[int, int] | None = None,
    placebo_periods: int = 0,
    cov_type: str = "conley",
    cutoff_km: float | None = None,
    time_lags: int | None = None,
    kernel: str = "bartlett",
    cluster_col: str | None = None,
    small_sample: bool = True,
    psd_adjust: str = "warn",
    alpha: float = 0.05,
    max_params: int = 400,
) -> SpatialDiDResult:
    """Spillover-robust difference-in-differences with exposure rings.

    One coefficient per distance ring plus the effect on the treated, the naive
    (no-rings) estimate and the exact Frisch-Waugh contamination decomposition
    beside it, Conley spatial HAC inference, a per-ring event study, per-ring
    pre-trend and placebo tests, and the outermost-ring contrast test that is
    the only evidence the comparison group is clean.

    Parameters
    ----------
    df : pd.DataFrame
        Long panel, one row per ``(unit, time)``. The identifiers may be
        columns or the two levels of a MultiIndex.
    unit, time, outcome, treat_time : str
        Column names. ``treat_time`` is the per-unit first-treatment period,
        ``NaN`` for never-treated units, and must be constant within a unit.
        ``outcome`` must be finite on every row handed in: the panel may be
        unbalanced (rows simply absent), but a present row carrying ``NaN`` or
        ``inf`` raises rather than being dropped without saying so.
    coords : DataFrame or mapping, optional
        Unit coordinates: a DataFrame indexed by unit id whose first two
        columns are ``[lat, lon]`` (or ``[x, y]`` with ``metric='euclidean'``),
        or a mapping ``id -> (lat, lon)``. A bare array is **rejected** — its
        row order would have to be guessed against the panel's unit ids, and a
        wrong guess silently mis-assigns every distance. Use
        :func:`exposure_rings` with ``ids=`` and pass the result as
        ``assignment=`` if you only have an array.
    assignment : RingAssignment, optional
        A ring assignment built by :func:`exposure_rings` or
        :func:`contiguity_rings`. Passing one makes ``rings``, ``ring_timing``
        and ``metric`` meaningless, so setting them raises. A graph assignment
        forces ``cov_type='cluster'``.
    ring_col : str, optional
        Name of a column of ``df`` holding a precomputed integer ring code in
        ``0..R+1`` (``R = len(rings) - 1``). ``coords`` may be supplied
        alongside, and is then used only for the HAC. Nothing on this route
        computes a distance or a timing, so ``ring_timing`` raises, ``metric``
        raises unless ``cov_type='conley'`` is actually going to use it, and
        the result reports ``ring_timing = distance_unit = 'unknown'`` rather
        than echoing a setting that did nothing.
    rings : sequence of float, default (0.0, 25.0, 50.0, 100.0)
        Band cut points; must start at ``0.0`` and be strictly increasing.
    ring_timing : {'already_treated', 'ever_treated'}, default 'already_treated'
        Meaningless — and therefore an error — with ``assignment=`` or
        ``ring_col=``.
    metric : {'haversine', 'euclidean'}, default 'haversine'
        Distance metric for both the rings and the Conley kernel.
        ``'graph'`` is a :class:`RingAssignment` tag only and is not accepted
        here — hop counts have no kilometre scale for the HAC to use.
        Meaningless — and therefore an error — with ``assignment=``, or with
        ``ring_col=`` unless ``cov_type='conley'`` needs it for the kernel.
    include_not_yet_treated_in_rings : bool, default True
        Whether a unit that will be treated later counts as a spillover
        recipient meanwhile. ``False`` drops **only** the rows of not-yet-
        treated units whose current ring code lies in ``1..R``; every treated
        unit keeps its full pre-period, so ``delta_0`` stays identified.
    min_units_per_ring : int, default 1
        Spillover rings carried by fewer units than this have their **rows**
        dropped (never reassigned to the control group, which would contaminate
        it) and are flagged ``dropped`` in the ring table.
    event_study : bool, default True
    event_study_estimator : {'auto', 'twfe', 'sunab'}, default 'auto'
        ``'auto'`` resolves to ``'twfe'`` with one cohort and ``'sunab'``
        otherwise. This governs **only** the event study: the ring table and
        every headline number always come from the static two-way-FE fit.
    event_window : tuple of int, optional
        ``(e_min, e_max)`` with ``e_min <= -1 <= e_max``; endpoints are binned
        into two catch-all cells. Read only by the event study, so it raises
        with ``event_study=False``.
    placebo_periods : int, default 0
        Refit the static specification on the pre-exposure subsample with a
        fake entry ``placebo_periods`` earlier.
    cov_type : {'conley', 'cluster'}, default 'conley'
        Conley (1999) space-time HAC, or cluster-robust by unit. Cluster
        under-covers here; see the module docstring.
    cutoff_km : float, **required** under ``cov_type='conley'``
        Conley distance cutoff, in the units of the coordinates. There is no
        default: the radius is an assumption about how far the error field
        reaches, and silently guessing it would bury that choice. ``2 *
        rings[-1]`` is the natural starting point — two untreated units can
        each sit within ``rings[-1]`` of the same treated unit and still be
        ``2 * rings[-1]`` apart, so every correlation the ring design itself
        posits is inside that radius — and the error message says so. ``0`` is
        legal and gives HC0. Meaningless, and an error, under
        ``cov_type='cluster'``.
    time_lags : int, optional
        Bartlett bandwidth in time. ``None`` uses
        ``max(1, floor(4 (T/100)^(2/9)))``, forced to 0 when ``T < 5``.
    kernel : {'bartlett', 'uniform'}, default 'bartlett'
    cluster_col : str, optional
        Cluster variable for ``cov_type='cluster'``; the unit by default.
    small_sample : bool, default True
        Apply the finite-sample multiplier documented on ``dof_factor``.
    psd_adjust : {'none', 'warn', 'clip'}, default 'warn'
        What to do when the HAC meat is indefinite. ``'clip'`` projects it onto
        the PSD cone, which is **conservative**: every variance weakly
        increases, so the clipped standard errors are never smaller. The
        cluster meat is a sum of outer products and therefore PSD by
        construction, so anything but ``'warn'`` raises under
        ``cov_type='cluster'``.
    alpha : float, default 0.05
        Two-sided level; critical values are standard normal.
    max_params : int, default 400
        Cap on the number of event-study dummies. Read only by the event
        study, so it raises with ``event_study=False``.

    Returns
    -------
    SpatialDiDResult

    Raises
    ------
    ValueError
        Missing columns; ``treat_time`` varying within a unit; ``rings``
        malformed; none / more than one of ``coords`` / ``assignment`` /
        ``ring_col``; ring codes outside ``0..R+1``; unknown
        ``ring_timing`` / ``metric`` / ``cov_type`` / ``kernel`` /
        ``psd_adjust`` / ``event_study_estimator``; ``alpha`` outside
        ``(0, 1)``; fewer than two periods; a constant outcome; a non-numeric
        or non-finite (NaN / inf) outcome column; no treated
        units; **no clean
        control group**; no exposed observations; no pre-period; a ring column
        that is constant within every unit carrying it; ``time_lags`` negative
        or ``>= n_periods``; ``cutoff_km`` negative, or omitted under
        ``cov_type='conley'``; ``max_params`` exceeded;
        ``metric='graph'`` or a graph assignment with ``cov_type='conley'``;
        ``cov_type='conley'`` without coordinates; a generated column name
        colliding with the user's. Also every keyword that is meaningless for
        the mode it was passed in: ``cluster_col`` outside
        ``cov_type='cluster'``; ``cutoff_km`` / ``time_lags`` / ``kernel``
        under ``cov_type='cluster'``; ``psd_adjust`` under
        ``cov_type='cluster'``; ``event_study_estimator`` / ``event_window`` /
        ``max_params`` with ``event_study=False``; ``rings`` / ``ring_timing``
        / ``metric`` with ``assignment=``; ``ring_timing`` (and ``metric``
        outside a Conley fit) with ``ring_col=``.
    KeyError
        ``coords`` missing one or more panel units; an assignment that does not
        cover the panel grid.
    TypeError
        Any unrecognised keyword — there is no ``**kwargs`` sink.

    Warns
    -----
    RuntimeWarning
        Dropped or tiny rings; ``cutoff_km < rings[-1]``; more than 10% of
        exposed units switching ring; an indefinite HAC meat; a non-positive
        sandwich variance; fewer than ten spatially separated treated clusters;
        ``T < 5`` forcing ``time_lags = 0``; a cohort with no ``e = -1``
        reference; an unusable placebo sample.

    Notes
    -----
    The Frisch-Waugh identity ``naive_att = direct_effect + theta @ coef[1:]``
    holds to machine precision, but only inside the one within projection this
    function runs. When rows are dropped (``min_units_per_ring > 1``,
    ``include_not_yet_treated_in_rings=False``) ``naive_att`` is the naive DiD
    on the *estimation* sample, not on the untrimmed panel; ``n_obs_dropped``
    says how far apart the two samples are and ``summary()`` prints it.

    Examples
    --------
    A single-cohort planar panel with a decaying spillover. The call below is
    the plainest one this function accepts, and runs as written — ``cutoff_km``
    has no default because the Conley radius is an assumption you have to make
    yourself:

    >>> import numpy as np, pandas as pd
    >>> from puremacro.did.spatial_did import spatial_did
    >>> rng = np.random.default_rng(0)
    >>> n, T, t0 = 90, 8, 4
    >>> x = rng.uniform(0, 300, n); yq = rng.uniform(0, 300, n)
    >>> xy = pd.DataFrame({'x': x, 'y': yq}, index=[f'u{i:02d}' for i in range(n)])
    >>> treated = xy.index[:6]
    >>> tt = pd.Series({u: (t0 if u in set(treated) else np.nan) for u in xy.index})
    >>> d = np.sqrt(((xy.to_numpy()[:, None, :]
    ...               - xy.loc[treated].to_numpy()[None, :, :]) ** 2).sum(-1)
    ...             ).min(axis=1)
    >>> eff = np.where(d == 0, 1.0, np.where(d <= 25, 0.5,
    ...                np.where(d <= 50, 0.25, np.where(d <= 100, 0.1, 0.0))))
    >>> rows = []
    >>> mu = rng.normal(size=n); tau = rng.normal(size=T) * 0.2
    >>> for i, u in enumerate(xy.index):
    ...     for t in range(T):
    ...         rows.append({'unit': u, 'time': t,
    ...                      'y': mu[i] + tau[t] + eff[i] * (t >= t0)
    ...                           + 0.15 * rng.normal(),
    ...                      'treat_time': tt[u]})
    >>> panel = pd.DataFrame(rows)
    >>> res = spatial_did(panel, coords=xy, metric='euclidean',
    ...                   cutoff_km=200.0)
    >>> np.round(res.coef, 2)
    array([1.03, 0.58, 0.27, 0.12])
    >>> bool(res.naive_att < res.direct_effect)
    True
    >>> float(np.abs(res.naive_att - (res.direct_effect
    ...                               + res.theta @ res.coef[1:]))) < 1e-12
    True
    """
    func = "spatial_did"
    # ---------------- STEP 0: validate -------------------------------------
    metric_given = metric is not _UNSET
    ring_timing_given = ring_timing is not _UNSET
    if not metric_given:
        metric = "haversine"
    if not ring_timing_given:
        ring_timing = "already_treated"
    if ring_timing not in _RING_TIMINGS:
        raise ValueError(
            f"{func}: ring_timing must be one of {_RING_TIMINGS}, "
            f"got {ring_timing!r}")
    if metric not in _METRICS:
        raise ValueError(
            f"{func}: metric must be one of {_METRICS}, got {metric!r}"
            + (" — 'graph' is a RingAssignment tag only; build the rings with "
               "contiguity_rings and pass them as assignment= with "
               "cov_type='cluster'" if metric == "graph" else ""))
    if cov_type not in _COV_TYPES:
        raise ValueError(
            f"{func}: cov_type must be one of {_COV_TYPES}, got {cov_type!r}")
    if kernel not in _KERNELS:
        raise ValueError(
            f"{func}: kernel must be one of {_KERNELS}, got {kernel!r}")
    if psd_adjust not in _PSD_ADJUST:
        raise ValueError(
            f"{func}: psd_adjust must be one of {_PSD_ADJUST}, "
            f"got {psd_adjust!r}")
    if event_study_estimator not in _ES_ESTIMATORS:
        raise ValueError(
            f"{func}: event_study_estimator must be one of {_ES_ESTIMATORS}, "
            f"got {event_study_estimator!r}")
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"{func}: alpha must lie in (0, 1), got {alpha!r}")
    if int(min_units_per_ring) < 1:
        raise ValueError(
            f"{func}: min_units_per_ring must be >= 1, got {min_units_per_ring!r}")
    if int(placebo_periods) < 0:
        raise ValueError(
            f"{func}: placebo_periods must be >= 0, got {placebo_periods!r}")
    if int(max_params) < 1:
        raise ValueError(f"{func}: max_params must be >= 1, got {max_params!r}")
    if cutoff_km is not None and float(cutoff_km) < 0:
        raise ValueError(
            f"{func}: cutoff_km must be non-negative, got {cutoff_km!r}")
    if cluster_col is not None and cov_type != "cluster":
        raise ValueError(
            f"{func}: cluster_col is meaningless for cov_type={cov_type!r}; "
            "pass cov_type='cluster' or drop it")
    if event_window is not None:
        ew = tuple(int(v) for v in event_window)
        if len(ew) != 2 or ew[0] > -1 or ew[1] < 0:
            raise ValueError(
                f"{func}: event_window must be (e_min, e_max) with "
                f"e_min <= -1 <= e_max, got {event_window!r}")
        event_window = ew
    if not event_study:
        # G3: every keyword that does nothing in the chosen mode raises. The
        # event study is the only consumer of these three.
        dead = [nm for nm, live in (("event_study_estimator",
                                     event_study_estimator != "auto"),
                                    ("event_window", event_window is not None),
                                    ("max_params", int(max_params) != 400))
                if live]
        if dead:
            raise ValueError(
                f"{func}: {dead} are meaningless with event_study=False (only "
                "the event study reads them); drop them or pass "
                "event_study=True")
    if cov_type == "cluster" and psd_adjust != "warn":
        raise ValueError(
            f"{func}: psd_adjust={psd_adjust!r} is meaningless for "
            "cov_type='cluster' — the cluster meat is a sum of outer products "
            "and is positive semi-definite by construction, so it can never "
            "need clipping; drop it or pass cov_type='conley'")

    n_routes = sum(x is not None for x in (coords, assignment, ring_col))
    if n_routes == 0:
        raise ValueError(
            f"{func}: pass coords= (unit coordinates), assignment= (the output "
            "of exposure_rings / contiguity_rings) or ring_col= (a precomputed "
            "ring code column)")
    if assignment is not None and ring_col is not None:
        raise ValueError(
            f"{func}: pass either assignment= or ring_col=, not both")
    if assignment is not None:
        if not isinstance(assignment, RingAssignment):
            raise ValueError(
                f"{func}: assignment must be a RingAssignment, got "
                f"{type(assignment).__name__}")
        bad = ([nm for nm, v, dflt in
                (("rings", tuple(float(r) for r in rings),
                  (0.0, 25.0, 50.0, 100.0)),)
                if v != dflt]
               + [nm for nm, given in (("ring_timing", ring_timing_given),
                                       ("metric", metric_given)) if given])
        if bad:
            raise ValueError(
                f"{func}: {bad} are meaningless when assignment= is given (the "
                "assignment already fixes the bands, the timing and the "
                "metric); drop them")

    edges = _validate_edges(rings, func) if assignment is None else np.asarray(
        assignment.edges, dtype=float)
    R = len(edges) - 1

    need = [unit, time, outcome]
    if ring_col is None:
        need.append(treat_time)
    data = _long_frame(df, unit, time, need, func)
    if ring_col is not None and ring_col not in data.columns:
        raise ValueError(
            f"{func}: ring_col={ring_col!r} is not a column of the frame")
    has_treat_col = treat_time in data.columns

    data = data.sort_values([unit, time], kind="mergesort").reset_index(drop=True)
    if data.duplicated(subset=[unit, time]).any():
        dup = data.loc[data.duplicated(subset=[unit, time]), [unit, time]]
        raise ValueError(
            f"{func}: the panel has {len(dup)} duplicated (unit, time) row(s), "
            f"e.g. {dup.head(3).to_dict('records')}")
    units = pd.Index(sorted(set(data[unit])), tupleize_cols=False)
    periods = pd.Index(sorted(set(data[time])), tupleize_cols=False)
    n_u, n_t = len(units), len(periods)
    if n_t < 2:
        raise ValueError(
            f"{func}: a difference-in-differences needs at least two periods, "
            f"got {n_t}")
    upos = {u: i for i, u in enumerate(units)}
    tpos = {t: i for i, t in enumerate(periods)}
    row_u = data[unit].map(upos).to_numpy(dtype=int)
    row_t = data[time].map(tpos).to_numpy(dtype=int)

    tt_vals = None
    if has_treat_col:
        nun = data.groupby(unit, sort=False)[treat_time].nunique(dropna=False)
        bad_units = list(nun[nun > 1].index[:5])
        if bad_units:
            raise ValueError(
                f"{func}: treat_time must be constant within a unit; it varies "
                f"for {int((nun > 1).sum())} unit(s), e.g. {bad_units}")
        firsts = data.drop_duplicates(subset=[unit]).set_index(unit)[treat_time]
        tt_vals = [firsts.loc[u] for u in units]
        if not (_entry_positions(periods, tt_vals, func) < n_t).any():
            raise ValueError(
                f"{func}: no treated units — every {treat_time!r} is NaN, or "
                "every treatment date falls outside the observed periods")

    # ---------------- STEP 1: ring assignment ------------------------------
    ra = None
    if assignment is not None:
        ra = assignment
        code_full = _codes_aligned(ra, units, periods, func)
        eff_metric, dist_unit = ra.metric, ra.distance_unit
        labels = tuple(ra.labels)
        ring_timing = ra.ring_timing
    elif ring_col is not None:
        # G3 / G11. On this route the codes come from the user's column, so
        # `ring_timing` never runs and `metric` only survives as the Conley
        # kernel's metric. Neither may be echoed back as if it had built the
        # rings: `ring_timing` and `distance_unit` are reported as 'unknown'.
        dead = [nm for nm, given in
                (("ring_timing", ring_timing_given),
                 ("metric", metric_given and cov_type != "conley")) if given]
        if dead:
            raise ValueError(
                f"{func}: {dead} are meaningless on the ring_col= route — the "
                "ring codes come from your column, so nothing here computes a "
                "distance or a timing; drop them, or pass coords= / "
                "assignment= and let this function build the rings")
        raw = data[ring_col].to_numpy()
        if not np.all(np.isfinite(np.asarray(raw, dtype=float))):
            raise ValueError(f"{func}: ring_col={ring_col!r} contains NaN")
        codes = np.asarray(raw, dtype=float)
        if not np.allclose(codes, np.round(codes)):
            raise ValueError(
                f"{func}: ring_col={ring_col!r} must hold integer ring codes")
        codes = codes.astype(np.int64)
        illegal = sorted(set(codes[(codes < 0) | (codes > R + 1)].tolist()))
        if illegal:
            raise ValueError(
                f"{func}: ring_col={ring_col!r} has code(s) {illegal} outside "
                f"0..{R + 1} (R = len(rings) - 1 = {R})")
        code_full = np.full((n_u, n_t), -1, dtype=np.int64)
        code_full[row_u, row_t] = codes
        eff_metric = metric
        dist_unit = "unknown"
        labels = _default_labels(edges, dist_unit)
        ring_timing = "unknown"
    else:
        if isinstance(coords, pd.DataFrame):
            missing = [u for u in units if u not in coords.index]
            if missing:
                raise KeyError(
                    f"{func}: coords is missing {len(missing)} unit(s), "
                    f"e.g. {missing[:3]}")
            coords_sub = coords.loc[list(units)]
        elif isinstance(coords, Mapping):
            missing = [u for u in units if u not in coords]
            if missing:
                raise KeyError(
                    f"{func}: coords is missing {len(missing)} unit(s), "
                    f"e.g. {missing[:3]}")
            coords_sub = pd.DataFrame(
                [list(coords[u])[:2] for u in units],
                index=list(units), columns=["c1", "c2"])
        else:
            raise ValueError(
                f"{func}: coords must be a DataFrame indexed by unit id or a "
                "mapping id -> (lat, lon); a bare array cannot be aligned to "
                "the panel's unit ids without guessing. Build the rings with "
                "exposure_rings(..., ids=...) and pass assignment= instead")
        ra = exposure_rings(
            coords_sub, treat_time=pd.Series(tt_vals, index=list(units)),
            times=list(periods), rings=edges, ring_timing=ring_timing,
            metric=metric)
        code_full = _codes_aligned(ra, units, periods, func)
        eff_metric, dist_unit = ra.metric, ra.distance_unit
        labels = tuple(ra.labels)

    if eff_metric == "graph" and cov_type == "conley":
        raise ValueError(
            f"{func}: a graph (hop-count) ring assignment has no kilometre "
            "scale for a Conley kernel; pass cov_type='cluster', or build the "
            "rings from coordinates with exposure_rings")
    if cov_type == "conley" and cutoff_km is None:
        raise ValueError(
            f"{func}: cov_type='conley' needs an explicit cutoff_km. The "
            "Conley radius is an assumption about how far the error field "
            "reaches, not something the ring design implies; resolving it "
            "silently to 2*rings[-1] would bury that choice in a default. "
            f"Pass cutoff_km={_fmt_edge(2.0 * float(edges[-1]))} "
            "(= 2*rings[-1], the widest separation two units in the same "
            "treated unit's outermost ring can have), cutoff_km=0.0 for HC0, "
            "or switch to cov_type='cluster'")

    # Only cells that carry a usable observation can be estimated from, and a
    # non-finite outcome is not one. Like spatial_panel and var.gvar, this
    # function refuses the column by name rather than thinning the estimation
    # sample behind the caller's back; an inf would otherwise reach the HAC and
    # die inside LAPACK with no function name on the error.
    y_col = data[outcome]
    try:
        y_vals = y_col.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{func}: the outcome column {outcome!r} is not numeric (dtype "
            f"{y_col.dtype}); it must hold finite floats") from exc
    n_bad = int(np.sum(~np.isfinite(y_vals)))
    if n_bad:
        raise ValueError(
            f"{func}: the outcome column {outcome!r} has {n_bad} non-finite "
            f"value(s) (NaN or inf) among the {len(data)} panel rows; drop or "
            "impute them first. Rows absent from the frame are fine — the "
            "panel may be unbalanced — but a present row with no usable "
            "outcome is refused rather than silently dropped, so that n_obs "
            "and n_obs_dropped mean what they say.")
    observed = np.zeros((n_u, n_t), dtype=bool)
    observed[row_u, row_t] = True
    code_full = np.where(observed, code_full, -1)

    if not (code_full == 0).any():
        raise ValueError(
            f"{func}: no treated units — every treat_time is NaN, or every "
            "treatment date falls outside the observed periods")

    code = code_full.copy()
    if not include_not_yet_treated_in_rings:
        if tt_vals is None:
            raise ValueError(
                f"{func}: include_not_yet_treated_in_rings=False needs the "
                f"{treat_time!r} column to know who is not yet treated")
        entry_pos = _entry_positions(periods, tt_vals, func)
        has_g = np.array([not _is_never(v) for v in tt_vals])
        not_yet = entry_pos[:, None] > np.arange(n_t)[None, :]
        drop = has_g[:, None] & not_yet & (code >= 1) & (code <= R)
        code[drop] = -1

    dropped_rings = []
    for r in range(1, R + 1):
        m = code == r
        nu = int(m.any(axis=1).sum())
        if 0 < nu < int(min_units_per_ring):
            warnings.warn(
                f"{func}: ring {r} ({labels[r]}) is carried by {nu} unit(s), "
                f"below min_units_per_ring={min_units_per_ring}; its rows are "
                "dropped from the sample (never moved into the control group)",
                RuntimeWarning, stacklevel=2)
            code[m] = -1

    # ---------------- STEP 2: identification gates -------------------------
    exposed_cell = (code >= 0) & (code <= R)
    control_cell = code == R + 1
    if not exposed_cell.any():
        raise ValueError(f"{func}: no exposed observations to estimate from")
    both = exposed_cell.any(axis=0) & control_cell.any(axis=0)
    if not both.any():
        raise ValueError(
            f"{func}: no clean control group — no period contains both an "
            "exposed observation and one beyond the outermost ring "
            f"(rings[-1] = {_fmt_edge(edges[-1])} {dist_unit}). The ring "
            "indicators then sum to a function of the period alone and are "
            "collinear with the time fixed effects. Widen the sample or "
            "shrink `rings`.")

    has_entry, a_idx, entry_ring, _ev, switch = _event_clock(code, R)
    if has_entry.any() and np.all(a_idx[has_entry] == 0):
        raise ValueError(
            f"{func}: no pre-period — every exposed unit is already exposed in "
            "the first observed period, so its ring indicator is constant "
            "within the unit and absorbed by the unit fixed effect")

    kept = []
    for r in range(R + 1):
        if (code == r).any():
            kept.append(r)
        else:
            if r == 0:
                raise ValueError(
                    f"{func}: no treated unit-period survives the exclusion "
                    "rules, so the direct effect is not identified")
            dropped_rings.append(r)
            warnings.warn(
                f"{func}: ring {r} ({labels[r]}) has no unit-period and is "
                "dropped from the design; the estimator reduces to the "
                "two-way-FE DiD over the remaining rings",
                RuntimeWarning, stacklevel=2)
    dropped_rings = tuple(sorted(set(dropped_rings)))

    valid = code != -1
    for r in kept:
        col = code == r
        carriers = np.flatnonzero(col.any(axis=1))
        varies = False
        for i in carriers:
            cells = col[i][valid[i]]
            if cells.size and not cells.all():
                varies = True
                break
        if not varies:
            hint = ("; include_not_yet_treated_in_rings=False removed the "
                    "rows that gave it variation" if not
                    include_not_yet_treated_in_rings else "")
            raise ValueError(
                f"{func}: the ring-{r} indicator ({labels[r]}) is constant "
                "within every unit that carries it, so the unit fixed effects "
                f"absorb it and delta_{r} is not identified{hint}")

    n_switchers = int(switch.sum())
    n_exposed = int(has_entry.sum())
    if n_exposed and n_switchers > 0.10 * n_exposed:
        warnings.warn(
            f"{func}: {n_switchers} of {n_exposed} exposed units change ring "
            "after entry; the static model uses the CURRENT ring while the "
            "event study anchors its clock at entry, so the two answer "
            "slightly different questions",
            RuntimeWarning, stacklevel=2)

    # ---------------- STEP 3: the static long fit --------------------------
    keep_row = valid[row_u, row_t]
    code_row = code[row_u, row_t]
    cols, xnames = _build_design(code_row[keep_row], kept, labels,
                                 list(data.columns), func)
    est = data.loc[keep_row, [unit, time, outcome]]
    if est[outcome].nunique(dropna=True) <= 1:
        raise ValueError(
            f"{func}: the outcome {outcome!r} has no variation in the "
            "estimation sample, so nothing is identified")
    est = pd.concat(
        [est, pd.DataFrame(cols, index=est.index)], axis=1)
    n_obs_dropped = int(keep_row.size - keep_row.sum())

    panel, ent_lv, tim_lv = as_panel_index(
        est, entity_level=unit, time_level=time, unit_col=unit, time_col=time,
        func=func)
    fit = two_way_fe_within(panel, y_col=outcome, x_cols=xnames,
                            entity_level=ent_lv, time_level=tim_lv)
    Xw, yw = fit["X_within"], fit["y_within"]
    beta = fit["beta"]
    ent_keys = np.asarray(fit["entity_keys"])
    tim_keys = np.asarray(fit["time_keys"])
    n_obs = int(fit["n_obs"])
    n_units_fit = int(len(np.unique(ent_keys)))
    n_periods_fit = int(len(np.unique(tim_keys)))

    # ---------------- STEP 6a: covariance settings -------------------------
    coords_df = None
    if coords is not None:
        if isinstance(coords, pd.DataFrame):
            coords_df = coords.iloc[:, :2]
        elif isinstance(coords, Mapping):
            coords_df = pd.DataFrame(
                [list(coords[u])[:2] for u in units], index=list(units),
                columns=["c1", "c2"])
        else:
            raise ValueError(
                f"{func}: coords must be a DataFrame indexed by unit id or a "
                "mapping id -> (lat, lon)")
    if cov_type == "conley" and coords_df is None:
        raise ValueError(
            f"{func}: cov_type='conley' needs unit coordinates; pass coords= "
            "or switch to cov_type='cluster'")

    cut = None
    lags = None
    n_clusters = None
    if cov_type == "conley":
        cut = float(cutoff_km)
        if cut > 0 and cut < edges[-1]:
            warnings.warn(
                f"{func}: cutoff_km={_fmt_edge(cut)} is below rings[-1]="
                f"{_fmt_edge(edges[-1])}, so the HAC treats as independent "
                "units that the ring design itself says are spillover-linked",
                RuntimeWarning, stacklevel=2)
        if time_lags is None:
            lags = _dk_bandwidth(n_periods_fit)
            if n_periods_fit < 5:
                warnings.warn(
                    f"{func}: only {n_periods_fit} periods; time_lags is forced "
                    "to 0 (a Bartlett bandwidth needs a time dimension to run "
                    "along)",
                    RuntimeWarning, stacklevel=2)
                lags = 0
        else:
            lags = int(time_lags)
            if lags < 0:
                raise ValueError(
                    f"{func}: time_lags must be non-negative, got {time_lags!r}")
            if lags >= n_periods_fit:
                raise ValueError(
                    f"{func}: time_lags={lags} must be below the number of "
                    f"periods ({n_periods_fit})")
        treated_units = np.flatnonzero((code_full == 0).any(axis=1))
        tc = coords_df.loc[[units[i] for i in treated_units]].to_numpy(float)
        n_clusters = _treated_cluster_count(tc, cut, eff_metric)
        if n_clusters < 10:
            warnings.warn(
                f"{func}: only {n_clusters} spatially separated treated "
                f"cluster(s) at a {_fmt_edge(cut)} {dist_unit} cutoff; Conley "
                "inference behaves like cluster-robust inference with that "
                "many clusters, so the normal critical values are optimistic",
                RuntimeWarning, stacklevel=2)
    else:
        if cutoff_km is not None or time_lags is not None or kernel != "bartlett":
            offend = [nm for nm, v in (("cutoff_km", cutoff_km),
                                       ("time_lags", time_lags),
                                       ("kernel", kernel if kernel != "bartlett"
                                        else None)) if v is not None]
            raise ValueError(
                f"{func}: {offend} are meaningless for cov_type='cluster'; drop "
                "them or pass cov_type='conley'")

    cluster_keys = None
    if cov_type == "cluster" and cluster_col is not None:
        if cluster_col not in data.columns:
            raise ValueError(
                f"{func}: cluster_col={cluster_col!r} is not a column of the "
                "frame")
        cl = data.loc[keep_row, [unit, time, cluster_col]].copy()
        cl = cl.set_index([unit, time])[cluster_col]
        cluster_keys = cl.loc[list(zip(ent_keys, tim_keys))].to_numpy()

    cov_kwargs = dict(
        cov_type=cov_type, coords_df=coords_df, cutoff_km=cut, time_lags=lags or 0,
        kernel=kernel, metric=eff_metric, cluster_keys=cluster_keys,
        small_sample=small_sample, psd_adjust=psd_adjust)
    cov = _sandwich(fit, func=func, **cov_kwargs)
    V = cov["vcov"]
    se = cov["se"]
    z = norm.ppf(1 - alpha / 2)

    # ---------------- STEP 4: naive estimate and Frisch-Waugh --------------
    X1 = Xw[:, :1]
    X2 = Xw[:, 1:]
    XtX1_inv = inv_xtx(X1, name=f"{func} (naive DiD)")
    b_naive = float((XtX1_inv @ X1.T @ yw)[0])
    u_naive = yw - X1 @ np.array([b_naive])
    theta = (XtX1_inv @ X1.T @ X2).ravel() if X2.shape[1] else np.zeros(0)
    naive_fit = {"X_within": X1, "residuals": u_naive, "XtX_inv": XtX1_inv,
                 "entity_keys": ent_keys, "time_keys": tim_keys}
    cov_naive = _sandwich(naive_fit, func=func, quiet=True, **cov_kwargs)
    naive_se = float(cov_naive["se"][0])

    c_vec = np.concatenate([[0.0], theta])
    contamination = float(c_vec @ beta)
    cvar = float(c_vec @ V @ c_vec)
    contamination_se = float(np.sqrt(cvar)) if cvar >= 0 else float("nan")

    # ---------------- STEP 7: derived quantities ---------------------------
    # N_0 is the number of EVER-TREATED units, not the number whose entry ring
    # happens to be 0. Under staggered adoption with clustered treatment a
    # later cohort's unit first appears as a spillover recipient, so its entry
    # ring is 1..R; counting it as a spillover recipient AND leaving it out of
    # the denominator inflated every N_r / N_0 ratio. Each unit is now counted
    # exactly once: treated units in N_0, never-treated exposed units in the
    # N_r of their entry ring.
    ever_treated = (code_full == 0).any(axis=1)
    n0 = int(ever_treated.sum())
    a_vec = np.zeros(len(kept))
    for j, r in enumerate(kept):
        if r == 0:
            a_vec[j] = 1.0
        else:
            nr = int(np.sum(has_entry & ~ever_treated & (entry_ring == r)))
            a_vec[j] = nr / n0 if n0 else np.nan
    total_effect = float(a_vec @ beta)
    tvar = float(a_vec @ V @ a_vec)
    total_effect_se = float(np.sqrt(tvar)) if tvar >= 0 else float("nan")

    names = tuple(_ring_name(r, labels[r]) for r in kept)
    lo_edge = {0: np.nan, R + 1: float(edges[-1])}
    hi_edge = {0: np.nan, R + 1: np.inf}
    for r in range(1, R + 1):
        lo_edge[r] = float(edges[r - 1])
        hi_edge[r] = float(edges[r])

    rows = []
    for r in range(R + 1):
        row = {"ring": r, "label": labels[r], "lo_edge": lo_edge[r],
               "hi_edge": hi_edge[r],
               "n_units": int((code_full == r).any(axis=1).sum()),
               "n_obs": int((code == r).sum())}
        if r in kept:
            j = kept.index(r)
            row.update(_inference_row(float(beta[j]), float(se[j]), z))
            row["dropped"] = False
        else:
            row.update({"effect": np.nan, "se": np.nan, "t": np.nan,
                        "p": np.nan, "lo": np.nan, "hi": np.nan})
            row["dropped"] = True
        rows.append(row)
    rows.append({"ring": R + 1, "label": labels[R + 1], "lo_edge": lo_edge[R + 1],
                 "hi_edge": hi_edge[R + 1],
                 "n_units": int((~has_entry).sum()),
                 "n_obs": int((code == R + 1).sum()),
                 "effect": np.nan, "se": np.nan, "t": np.nan, "p": np.nan,
                 "lo": np.nan, "hi": np.nan, "dropped": False})
    ring_table = pd.DataFrame(rows)[
        ["ring", "label", "lo_edge", "hi_edge", "n_units", "n_obs", "effect",
         "se", "t", "p", "lo", "hi", "dropped"]]

    units_per_ring = {int(r): int((code_full == r).any(axis=1).sum())
                      for r in range(R + 2)}
    small = {r: units_per_ring[r] for r in kept
             if units_per_ring[r] < _MIN_UNITS_FOR_JOINT}
    if small:
        warnings.warn(
            f"{func}: ring(s) {sorted(small)} are carried by fewer than "
            f"{_MIN_UNITS_FOR_JOINT} units ({small}); their per-coefficient "
            "standard errors are still fine, but every JOINT Wald statistic "
            "that touches such a ring is badly over-sized, so those rows are "
            "reported with a NaN statistic and underpowered=True rather than a "
            "confident p-value (see the module docstring for the measured "
            "rejection rates)",
            RuntimeWarning, stacklevel=2)

    def _gated(rings_touched: Sequence[int]) -> bool:
        """True when a joint restriction leans on an under-populated ring."""
        return any(units_per_ring.get(int(r), 0) < _MIN_UNITS_FOR_JOINT
                   for r in rings_touched)

    spill_idx = [j for j, r in enumerate(kept) if r >= 1]
    test_rows = []
    outer_bound = float("nan")
    outer_share = float("nan")
    if spill_idx:
        j_out = spill_idx[-1]
        Rm = np.zeros((1, len(kept)))
        Rm[0, j_out] = 1.0
        stat, dfree, p = _wald(beta, V, Rm, name=func)
        # A single-coefficient Wald is correctly sized even on a tiny ring
        # (measured 0.08 at nominal 0.10 with 5-9 units), so it is never gated.
        test_rows.append({"test": "outer_ring",
                          "null": f"delta_{kept[j_out]} = 0",
                          "stat": stat, "df": dfree, "p": p,
                          "underpowered": False})
        if np.isfinite(se[j_out]):
            outer_bound = float(abs(beta[j_out]) + z * se[j_out])
            d0 = abs(float(beta[0]))
            outer_share = outer_bound / d0 if d0 >= _SHARE_FLOOR else float("nan")
        Rm = np.zeros((len(spill_idx), len(kept)))
        for i, j in enumerate(spill_idx):
            Rm[i, j] = 1.0
        stat, dfree, p = _wald(beta, V, Rm, name=func)
        gate = _gated([kept[j] for j in spill_idx])
        test_rows.append({"test": "no_spillover",
                          "null": "delta_1 = ... = delta_R = 0",
                          "stat": np.nan if gate else stat, "df": dfree,
                          "p": np.nan if gate else p,
                          "underpowered": gate})
        if len(spill_idx) >= 2:
            Rm = np.zeros((len(spill_idx) - 1, len(kept)))
            for i in range(len(spill_idx) - 1):
                Rm[i, spill_idx[i]] = 1.0
                Rm[i, spill_idx[i + 1]] = -1.0
            stat, dfree, p = _wald(beta, V, Rm, name=func)
            test_rows.append({"test": "equal_rings",
                              "null": "delta_1 = ... = delta_R",
                              "stat": np.nan if gate else stat, "df": dfree,
                              "p": np.nan if gate else p,
                              "underpowered": gate})

    # ---------------- STEP 5 / 8: event study, pre-trends, placebo ---------
    ctx = {
        "code": code, "R": R, "kept": kept, "labels": labels,
        "has_entry": has_entry, "a_idx": a_idx, "entry_ring": entry_ring,
        "row_u": row_u[keep_row], "row_t": row_t[keep_row], "est": est,
        "unit": unit, "time": time, "outcome": outcome,
        "event_window": event_window, "max_params": int(max_params),
        "cov_kwargs": cov_kwargs, "alpha": float(alpha), "z": z,
        "lo_edge": lo_edge, "hi_edge": hi_edge, "gated": _gated,
    }
    es_df = att_ring_df = pre_df = None
    resolved_es = "none"
    n_cohorts = int(len(np.unique(a_idx[has_entry]))) if has_entry.any() else 0
    if event_study:
        resolved_es = (event_study_estimator if event_study_estimator != "auto"
                       else ("twfe" if n_cohorts <= 1 else "sunab"))
        ctx["estimator"] = resolved_es
        out = _event_study_fit(ctx)
        es_df, att_ring_df, pre_df = (out["event_study"], out["att_by_ring_es"],
                                      out["pretrend"])
        test_rows.extend(out["dynamics"])
    tests = pd.DataFrame(test_rows)[
        ["test", "null", "stat", "df", "p", "underpowered"]] \
        if test_rows else pd.DataFrame(
            columns=["test", "null", "stat", "df", "p", "underpowered"])

    placebo = None
    if int(placebo_periods) > 0:
        placebo = _placebo_fit(ctx, int(placebo_periods))

    n_treated_units = int((code_full == 0).any(axis=1).sum())
    n_control_units = int((~has_entry).sum())
    return SpatialDiDResult(
        ring_table=ring_table.copy(),
        coef=np.asarray(beta, dtype=float).copy(),
        vcov=np.asarray(V, dtype=float).copy(),
        names=names,
        ring_codes=tuple(int(r) for r in kept),
        naive_att=b_naive,
        naive_se=naive_se,
        direct_effect=float(beta[0]),
        direct_se=float(se[0]),
        contamination=contamination,
        contamination_se=contamination_se,
        theta=np.asarray(theta, dtype=float).copy(),
        total_effect=total_effect,
        total_effect_se=total_effect_se,
        event_study=None if es_df is None else es_df.copy(),
        att_by_ring_es=None if att_ring_df is None else att_ring_df.copy(),
        pretrend=None if pre_df is None else pre_df.copy(),
        placebo=None if placebo is None else placebo.copy(),
        tests=tests.copy(),
        outer_ring_contrast_bound=outer_bound,
        outer_ring_contrast_share=outer_share,
        ring_edges=tuple(float(e) for e in edges),
        ring_labels=tuple(labels),
        distance_unit=dist_unit,
        ring_timing=ring_timing,
        static_estimator="twfe",
        event_study_estimator=resolved_es,
        cov_type=cov_type,
        cutoff_km=cut,
        time_lags=lags,
        kernel=kernel,
        metric=eff_metric,
        psd_adjust=psd_adjust,
        include_not_yet_treated_in_rings=bool(include_not_yet_treated_in_rings),
        min_units_per_ring=int(min_units_per_ring),
        small_sample=bool(small_sample),
        alpha=float(alpha),
        n_obs=n_obs,
        n_obs_dropped=n_obs_dropped,
        n_units=n_units_fit,
        n_periods=n_periods_fit,
        n_treated_units=n_treated_units,
        n_control_units=n_control_units,
        n_cohorts=n_cohorts,
        n_ring_switchers=n_switchers,
        n_treated_clusters=n_clusters,
        meat_min_eig=cov["meat_min_eig"],
        n_negative_variance=int(cov["n_negative_variance"]),
        dof_factor=cov["dof_factor"],
        k_total=cov["k_total"],
        dropped_rings=dropped_rings,
        assignment=ra,
    )
