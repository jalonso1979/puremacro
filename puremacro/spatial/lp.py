"""Spatial local projections: own, neighbour and own-plus-neighbour responses.

Two-way fixed-effects panel local projections (Jorda 2005) augmented with the
spatially lagged shock, so that a regional impulse response splits into a
DIRECT response to the unit's own shock and an INDIRECT response to the shock
its neighbours receive.  For each horizon ``h`` the estimating equation is::

    y_{i,t+h} - y_{i,t-1} = mu_i^h + tau_t^h
                            + beta_h x_{i,t}
                            + sum_p gamma_{p,h} (W^(p) x)_{i,t}
                            + (lags of y, x, W x and the controls)
                            + eps_{i,t+h}

with ``W^(1) = W`` the spatial weights and ``W^(p)`` the p-th order neighbour
matrix in the Anselin (1988) sense (see :func:`higher_order_weights`).

What the coefficients mean
--------------------------
``beta_h`` and ``gamma_{p,h}`` are estimated **after** a two-way within
transformation, so they are responses RELATIVE TO THE PERIOD MEAN.  The time
effect ``tau_t^h`` absorbs, by construction, every component of the response
that is common to all units in a period — including the response to the
aggregate part of the shock.  Consequently:

* ``direct``   = ``beta_h``: the extra response of a unit whose own shock is
  one unit above the period average, holding its neighbourhood fixed;
* ``indirect`` = ``c_1 gamma_{1,h}``: the extra response of a unit whose
  first-ring neighbours' shocks are ``c_1`` above the period average — and one
  such label PER ORDER (``indirect2`` for the second ring, and so on), with
  ``indirect_all = sum_p c_p gamma_{p,h}`` added whenever more than one order
  is estimated;
* ``total``    = ``direct + sum_p c_p gamma_{p,h}``: a CROSS-SECTIONAL
  CONTRAST — the response of a unit whose own and neighbour shocks are both
  above the period average, relative to a unit at the period average.

``total`` is **not** the response to a uniform unit shock hitting every unit,
and this module never claims that it is.  A uniform shock is exactly the
variation ``tau_t`` removes: within-transformed, the predicted response to a
uniform ``+1`` is ``gamma * (s_i - sbar)``, whose cross-unit mean is zero.
Estimates are literally unchanged (to 1e-16) when an arbitrary aggregate
component ``kappa * xbar_t`` is added to the data-generating process, so no
aggregate multiplier is identified here.  ``total`` equals the aggregate
response only under the additional, untestable assumption that the common
response is zero.  This is the point of Chodorow-Reich (2019); see the Notes
section of :func:`spatial_lp`.

Inference
---------
Cluster-by-entity, Driscoll-Kraay (1998) and Conley (1999) / Hsiang (2010)
space-time HAC, with the full CROSS-HORIZON covariance of the focal
coefficients retained so that the standard error of ``direct + indirect``
carries the off-diagonal term and cumulative responses get honest bands.  No
finite-sample correction is applied anywhere (this is what makes the
``spillover_orders=()`` specification reproduce :func:`puremacro.lp.panel_lp`
bit for bit), so every band here is an asymptotic one and is measurably too
narrow on a regional panel — see the size table under "Deliberately omitted"
for how much, and treat a marginal ``gamma`` accordingly.

These "direct / indirect" effects are reduced-form coefficients on own and
neighbour *shocks*.  They are NOT the LeSage-Pace partial derivatives of a
simultaneous SAR/SDM model (the ``(I - rho W)^-1`` multipliers).

Deliberately omitted
--------------------
* **The cross-horizon joint spillover Wald test.**  A chi2 reference for the
  stacked restriction ``gamma_{p,h} = 0 at every h`` is grossly oversized in
  exactly the panels this module targets: a 150-replication Monte Carlo with
  ``N = 20``, ``T = 50``, ``H = 6`` and a true ``gamma = 0`` rejected at
  **0.313** against a nominal 0.10 (0.187 at nominal 0.05).  A
  many-restriction Wald with an estimated HAC covariance on T ~ 50 periods
  does not have its nominal size, so the test does not ship.  Per-horizon
  inference and the covariance-aware standard error of ``direct + indirect``
  do ship.  Test one horizon at a time, or use ``.vcov`` to build your own
  fixed-b reference.

  **The per-horizon z-test that DOES ship is not exact either**, and its size
  is reported here rather than represented by one favourable number.  On the
  same design (``N = 20``, ``T = 50``, ``h = 0..5``, true ``gamma = 0``, 400
  replications, so a Monte Carlo standard error of 0.011 at a nominal 0.05)
  the ``gamma_h`` z-test rejected at::

      cov_type            nominal 0.05              nominal 0.10
                        h = 0   worst over h      h = 0   worst over h
      driscoll-kraay    0.083   0.110 (h = 3)     0.138   0.172 (h = 5)
      conley 1000 km    0.068   0.100 (h = 3)     0.138   0.152 (h = 2)
      cluster           0.090   0.090 (h = 0)     0.160   0.160 (h = 0)

  So the honest contrast with the joint test's 0.313 is 0.08-0.11 at a nominal
  0.05, not the single ``h = 0`` Driscoll-Kraay figure: the size drifts up
  with the horizon as the overlapping-window MA(h) dependence grows.  Doubling
  the cross-section to ``N = 40`` (300 replications) does NOT fix Driscoll-
  Kraay (0.080 at ``h = 0``, 0.113 worst), which is consistent as
  ``T -> infinity`` and not in ``N``; Conley does improve (0.053 / 0.083), and
  cluster-by-entity is the smallest here only because ``T`` is large relative
  to ``N`` — it is still the covariance whose independence assumption this
  regression denies.  Read a ``gamma`` whose p-value sits near the threshold as
  undecided.
* **Instrumental variables.**  ``x`` is treated as exogenous conditional on
  the two-way fixed effects and the included lags.  The regional designs this
  module is intended to support (Chodorow-Reich 2019; Auerbach, Gorodnichenko
  and Murphy 2020; Dupor et al. 2023) instrument regional spending with
  military procurement or Bartik/shift-share exposure; that counterpart is not
  implemented here.  See :func:`puremacro.bartik.shift_share_iv` for the
  exposure instrument itself.
* **The simultaneous SAR/SDM decomposition** (LeSage-Pace direct / indirect /
  total partial derivatives, ``spatial_lag_y`` contemporaneously): only
  *lagged* spatial lags of ``y`` are offered, because a contemporaneous
  ``W y`` is endogenous and needs the spatial ML/GMM machinery.
* **Unbalanced weights.**  ``W`` must be built on exactly the panel's entity
  set; sub-setting it here would silently change its row sums and hence every
  ``gamma``'s interpretation.
* **Finite-sample / fixed-b corrections** of any kind (see Inference above).

References
----------
Jorda, O. (2005). Estimation and inference of impulse responses by local
    projections. American Economic Review 95(1), 161-182.
Auerbach, A.J., Gorodnichenko, Y. and Murphy, D. (2020). Local fiscal
    multipliers and fiscal spillovers in the USA. IMF Economic Review 68,
    195-229.
Dupor, B., Karabarbounis, M., Kudlyak, M. and Mehkari, M.S. (2023). Regional
    consumption responses and the aggregate fiscal multiplier. Review of
    Economic Studies 90(6), 2982-3021.
Chodorow-Reich, G. (2019). Geographic cross-sectional fiscal spending
    multipliers: what have we learned? AEJ: Economic Policy 11(2), 1-34.
Ramey, V.A. and Zubairy, S. (2018). Government spending multipliers in good
    times and in bad. Journal of Political Economy 126(2), 850-901.
Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer.
LeSage, J. and Pace, R.K. (2009). Introduction to Spatial Econometrics. CRC.
Driscoll, J.C. and Kraay, A.C. (1998). Consistent covariance matrix estimation
    with spatially dependent panel data. Review of Economics and Statistics
    80(4), 549-560.
Conley, T.G. (1999). GMM estimation with cross sectional dependence. Journal
    of Econometrics 92(1), 1-45.
Hsiang, S.M. (2010). Temperatures and cyclones strongly associated with
    economic production in the Caribbean and Central America. PNAS 107(35).
Newey, W.K. and West, K.D. (1987). A simple, positive semi-definite,
    heteroskedasticity and autocorrelation consistent covariance matrix.
    Econometrica 55(3), 703-708.
"""
from __future__ import annotations

import warnings
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm

from ..inference.dk import driscoll_kraay
from ..lp._common import resolve_lp_kwargs
from ..lp._panel_helpers import as_panel_index, two_way_fe_within
from ..lp._results import LPResult
from .hac import (
    _CLUSTER_ALIASES,
    _CONLEY_ALIASES,
    _DK_ALIASES,
    _entity_coords,
    kernel_matrix,
    spatial_hac_panel_meat,
)
from .weights import SpatialWeights, pairwise_distances

__all__ = [
    "spatial_lp",
    "SpatialLPResult",
    "higher_order_weights",
    "spatial_lag_panel",
]


# --------------------------------------------------------------------------
# Named constants (recorded on the result so a reported number is reproducible)
# --------------------------------------------------------------------------
WEAK_ID_CORR = 0.999
"""Within-sample |corr(x, W x)| above which ``gamma`` is flagged as weakly
identified (a ``RuntimeWarning``; the value is always stored in
``SpatialLPResult.spillover['corr_own']``)."""

ZERO_COLUMN_TOL = 1e-12
"""Relative tolerance below which a spillover column counts as identically
zero (an all-island / zero ``W``), which is an error rather than a silent
regressor drop."""

ROW_SUM_TOL = 1e-8
"""Standard deviation of the row sums of ``W`` above which the weights are
reported as not row-standardised."""

# _CLUSTER_ALIASES / _DK_ALIASES / _CONLEY_ALIASES now live in
# :mod:`puremacro.spatial.hac` so that ``spatial_panel``'s ``vcov=`` and this
# module's ``cov_type=`` accept the same token vocabulary; they are re-exported
# here (via the import above) for backwards compatibility.


# ==========================================================================
# Spatial-weights helpers (candidates for hoisting into spatial/weights.py)
# ==========================================================================
def higher_order_weights(
    weights: SpatialWeights,
    order: int,
    *,
    zero_lower_orders: bool = True,
    row_standardize: bool | None = None,
) -> SpatialWeights:
    """The ``p``-th order neighbour matrix in the Anselin (1988) convention.

    The support of ``W^(p)`` is ``{j : j is reachable from i by a walk of
    exactly p steps}`` MINUS the diagonal and (when ``zero_lower_orders``)
    MINUS the union of the supports of ``W^(1), ..., W^(p-1)``.  The surviving
    entries are taken from the matrix power ``W**p`` and the rows are
    re-standardised when the input was row-standardised, so ``(W^(p) x)_i``
    keeps its "average p-th ring neighbour shock" reading.

    Subtracting the lower orders is not cosmetic.  ``W**2`` puts a lot of mass
    back on FIRST-order neighbours (the walk ``i -> j -> k`` with ``k`` itself
    a neighbour of ``i``): on a row-standardised ``knn(3)`` matrix with
    ``n = 30``, after only zeroing the diagonal, 47% of each ``W^(2)`` row's
    mass on average (86% at worst) sits on first-order neighbours, and
    ``corr(W x, W^(2) x)`` over random ``x`` is 0.46.  ``gamma_2`` would then
    double-count the first ring instead of measuring a second ring.  Zeroing
    the diagonal alone is also required: the raw ``W**2`` diagonal reaches
    0.33 on ``knn(3)``, which would fold the own effect into ``gamma_2``.

    Parameters
    ----------
    weights : SpatialWeights
        The first-order weights.
    order : int
        Neighbour order ``p >= 1``.  ``order=1`` is the identity on
        ``weights`` — except that ``row_standardize=True`` is still honoured
        (it standardises the input), and ``row_standardize=False`` on
        already-standardised weights raises rather than silently returning
        standardised ones.
    zero_lower_orders : bool, default True
        Remove the supports of every lower order.  ``False`` keeps the raw
        ``W**p`` pattern (diagonal still zeroed), which is collinear with the
        lower orders — use it only if you know why you want it.
    row_standardize : bool, optional
        Re-standardise the rows.  ``None`` (default) follows
        ``weights.row_standardized``.

    Returns
    -------
    SpatialWeights
        ``kind`` is ``f"{weights.kind}^{order}"`` (``weights`` itself, or its
        standardisation, at ``order=1``).

    Raises
    ------
    ValueError
        If ``order`` is not a positive integer --- including a non-numeric
        ``order``, which is reported by name rather than as a bare ``float()``
        failure --- or ``order=1`` is combined
        with ``row_standardize=False`` on already-standardised weights (which
        cannot be undone — G3 forbids ignoring the keyword instead).

    Notes
    -----
    The masks are dense ``(n, n)`` boolean arrays.  That is the package's
    deliberate choice for regional panels (see ``puremacro.spatial.hac``); at
    ``n = 3000`` the masks cost ~9 MB each.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.spatial import knn_weights
    >>> from puremacro.spatial.lp import higher_order_weights
    >>> coords = pd.DataFrame({'lat': np.linspace(30.0, 45.0, 10),
    ...                        'lon': np.zeros(10)}, index=list('ABCDEFGHIJ'))
    >>> W1 = knn_weights(coords, 2)
    >>> W2 = higher_order_weights(W1, 2)
    >>> bool(np.all(W2.W.diagonal() == 0))
    True
    >>> # the two supports are disjoint
    >>> bool((W1.W.astype(bool).multiply(W2.W.astype(bool))).nnz == 0)
    True
    """
    try:
        order_f = float(order)
    except (TypeError, ValueError):
        raise ValueError(
            f"higher_order_weights: order must be a positive integer, got "
            f"{order!r} of type {type(order).__name__}") from None
    if isinstance(order, bool) or not order_f.is_integer() or int(order_f) < 1:
        raise ValueError(
            f"higher_order_weights: order must be a positive integer, got {order!r}")
    order = int(order_f)
    if order == 1:
        if row_standardize is None or bool(row_standardize) is bool(weights.row_standardized):
            return weights
        if row_standardize:
            return weights.standardize()
        raise ValueError(
            "higher_order_weights: order=1 with row_standardize=False cannot "
            "un-standardise already row-standardised weights; rebuild them with "
            "row_standardize=False instead of asking for it here")
    n = weights.n
    W = sp.csr_matrix(weights.W, dtype=float)
    B = W.copy()
    B.data = np.ones_like(B.data)

    reach = B.copy()                       # walks of exactly q steps, q = 1
    lower = np.zeros((n, n), dtype=bool)   # union of the supports of orders < q
    Wp = W.copy()
    for _ in range(2, order + 1):
        lower |= reach.astype(bool).toarray()
        reach = (reach @ B).tocsr()
        reach.data = np.ones_like(reach.data)
        Wp = (Wp @ W).tocsr()

    mask = reach.astype(bool).toarray()
    if zero_lower_orders:
        mask &= ~lower
    np.fill_diagonal(mask, False)
    vals = Wp.toarray() * mask
    std = weights.row_standardized if row_standardize is None else bool(row_standardize)
    out = SpatialWeights(sp.csr_matrix(vals), weights.ids,
                         f"{weights.kind}^{order}", False)
    if std:
        out = out.standardize()
    if out.n_islands:
        warnings.warn(
            f"higher_order_weights: {out.n_islands} unit(s) have no order-{order} "
            f"neighbour, e.g. {list(out.islands)[:3]}; their order-{order} spillover "
            "regressor is identically zero and the effective population for that "
            "order is smaller than n",
            RuntimeWarning, stacklevel=2,
        )
    return out


def spatial_lag_panel(
    df: pd.DataFrame,
    columns: str | Sequence[str],
    weights: SpatialWeights,
    *,
    entity_level: str = "code",
    time_level: str = "date",
    prefix: str = "W_",
    suffix: str = "",
    unbalanced: str = "propagate",
) -> pd.DataFrame:
    """Per-period spatial lag ``(W x)_{i,t}`` of one or more panel columns.

    Parameters
    ----------
    df : pd.DataFrame
        Panel indexed by a two-level ``(entity, time)`` MultiIndex.
    columns : str or sequence of str
        Column(s) to lag.
    weights : SpatialWeights
        Weights whose ``ids`` are the panel entities (every entity must be an
        id and every id must be an entity).
    entity_level, time_level : str
        MultiIndex level names.
    prefix, suffix : str
        Output column names are ``prefix + col + suffix``.
    unbalanced : {'propagate', 'renormalize'}, default 'propagate'
        ``'propagate'`` lets a missing neighbour value poison the lag along
        non-zero weights only (a missing NON-neighbour never contaminates,
        because a sparse product never multiplies a structural zero by NaN),
        so a row with an incomplete neighbourhood drops from the regression.
        ``'renormalize'`` returns the observed-neighbour weighted mean rescaled
        back to the original row sum, ``NaN`` only when no neighbour is
        observed.  Island rows are ``0.0`` under both policies.

    Returns
    -------
    pd.DataFrame
        One column per entry of ``columns``, aligned on ``df.index``.

    Raises
    ------
    KeyError
        If a requested column is not in ``df``, if a weights id is absent from
        the panel, or if a panel entity is absent from the weights.  Both
        entity directions are checked: an entity with no row in ``W`` has no
        spatial lag at all, and returning a silent all-NaN column for it would
        drop it from every downstream regression without saying so.
    ValueError
        If ``unbalanced`` is not one of the two policies, or the index is not
        a two-level MultiIndex.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.spatial import contiguity_weights
    >>> from puremacro.spatial.lp import spatial_lag_panel
    >>> W = contiguity_weights({'a': ['b'], 'b': ['a', 'c'], 'c': ['b']})
    >>> idx = pd.MultiIndex.from_product([['a', 'b', 'c'], [0, 1]],
    ...                                  names=['code', 'date'])
    >>> df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=idx)
    >>> spatial_lag_panel(df, 'x', W)['W_x'].round(3).tolist()
    [3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    """
    if unbalanced not in ("propagate", "renormalize"):
        raise ValueError(
            "spatial_lag_panel: unbalanced must be 'propagate' or 'renormalize', "
            f"got {unbalanced!r}")
    if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels < 2:
        raise ValueError(
            "spatial_lag_panel: the panel needs a two-level (entity, time) MultiIndex")
    cols = [columns] if isinstance(columns, str) else list(columns)
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise KeyError(f"spatial_lag_panel: column(s) {absent} are not in the frame; "
                       f"columns are {list(df.columns)[:10]}")
    ids = list(weights.ids)
    entities = pd.Index(df.index.get_level_values(entity_level)).unique()
    ent_set = set(entities)
    missing = [u for u in ids if u not in ent_set]
    if missing:
        raise KeyError(
            f"spatial_lag_panel: {len(missing)} weights id(s) are absent from the "
            f"panel, e.g. {missing[:3]}")
    id_set = set(ids)
    extra = [u for u in entities if u not in id_set]
    if extra:
        raise KeyError(
            f"spatial_lag_panel: {len(extra)} panel entit(ies) are absent from the "
            f"weights, e.g. {extra[:3]}; their spatial lag is undefined (it would come "
            "back all-NaN). Rebuild the weights on the panel's entity set.")

    periods = pd.Index(df.index.get_level_values(time_level)).unique().sort_values()
    out_index = pd.MultiIndex.from_product([ids, list(periods)],
                                           names=[entity_level, time_level])
    Wm = weights.W
    rs = weights.row_sums()
    out = {}
    for col in cols:
        piv = df[col].unstack(entity_level).reindex(index=periods, columns=ids)
        M = piv.to_numpy(dtype=float)                      # (T, n)
        if unbalanced == "propagate":
            lag = np.asarray(Wm @ M.T)                     # (n, T)
        else:
            obs = np.isfinite(M)
            num = np.asarray(Wm @ np.nan_to_num(M, nan=0.0).T)
            den = np.asarray(Wm @ obs.astype(float).T)
            with np.errstate(invalid="ignore", divide="ignore"):
                lag = np.where(den > 0, num * rs[:, None] / np.where(den > 0, den, 1.0),
                               np.nan)
        lag = np.where(rs[:, None] == 0.0, 0.0, np.asarray(lag))
        ser = pd.Series(np.asarray(lag).ravel(), index=out_index)
        out[f"{prefix}{col}{suffix}"] = ser.reindex(df.index)
    return pd.DataFrame(out, index=df.index)


# ==========================================================================
# Covariance primitives (candidates for hoisting into lp/_panel_helpers.py)
# ==========================================================================
CovFn = Callable[[dict], np.ndarray]
"""Callable mapping the dict returned by :func:`two_way_fe_within` to the full
``k x k`` sandwich covariance of its coefficients."""


def _dk_bandwidth(time_keys: np.ndarray, time_lags: int | None = None) -> int:
    """Driscoll-Kraay Bartlett bandwidth ``floor(4 (T/100)^(2/9))`` on the
    number of PERIODS (never the number of panel rows)."""
    if time_lags is not None:
        return int(time_lags)
    T_dk = int(len(np.unique(time_keys)))
    return max(1, int(np.floor(4 * (T_dk / 100) ** (2 / 9))))


def _cluster_cov(out: dict) -> np.ndarray:
    """Cluster-robust-by-entity covariance (already inside
    :func:`two_way_fe_within`)."""
    return out["vcov"]


def _make_dk_cov_fn(time_lags: int | None = None) -> CovFn:
    """Driscoll-Kraay (1998) sandwich; arithmetic identical to
    ``puremacro.lp._panel_helpers._focal_dk_se`` when ``time_lags is None``."""

    def _cov(out: dict) -> np.ndarray:
        X = out["X_within"]
        u = out["residuals"]
        score = X * u[:, None]
        L = _dk_bandwidth(out["time_keys"], time_lags)
        S = driscoll_kraay(score, out["time_keys"], lags=L)
        XtX_inv = out["XtX_inv"]
        return XtX_inv @ S @ XtX_inv

    return _cov


def _make_conley_cov_fn(
    coords: Any,
    *,
    cutoff_km: float,
    time_lags: int | None = None,
    kernel: str = "bartlett",
    metric: str = "haversine",
) -> CovFn:
    """Conley (1999) / Hsiang (2010) space-time HAC sandwich; arithmetic
    identical to ``puremacro.lp._panel_helpers.make_conley_se_fn``."""

    def _cov(out: dict) -> np.ndarray:
        X = out["X_within"]
        u = out["residuals"]
        L = _dk_bandwidth(out["time_keys"], time_lags)
        S = spatial_hac_panel_meat(
            X, u, coords, out["entity_keys"], out["time_keys"],
            cutoff_km, L, kernel=kernel, metric=metric,
        )
        XtX_inv = out["XtX_inv"]
        return XtX_inv @ S @ XtX_inv

    return _cov


def _focal_influence(out: dict, focal: Sequence[int]) -> np.ndarray:
    """The ``(n_obs, |focal|)`` influence series ``psi[r, f] = (A[f, :] x_r) u_r``.

    Its kernel-weighted double sum IS the sandwich covariance of the focal
    coefficients: ``A[F, :] S A[F, :]' = sum_{r,s} K(r,s) psi_r psi_s'``
    because ``A = (X'X)^-1`` is symmetric.  This is what makes a cross-horizon
    covariance affordable: memory ``H * T * n * |F|`` instead of
    ``H * T * n * k``.
    """
    A = out["XtX_inv"]
    return (out["X_within"] @ A[list(focal), :].T) * out["residuals"][:, None]


def _panel_score_grid(
    U: np.ndarray,
    entity_keys: Any,
    time_keys: Any,
    entities: pd.Index,
    periods: pd.Index,
) -> np.ndarray:
    """Grid per-observation scores onto a common ``(T, n, k)`` axis, zero-filling
    absent entity-periods (the moment condition simply does not exist there)."""
    ent_pos = entities.get_indexer(pd.Index(np.asarray(entity_keys)))
    t_pos = periods.get_indexer(pd.Index(np.asarray(time_keys)))
    G = np.zeros((len(periods), len(entities), U.shape[1]))
    G[t_pos, ent_pos, :] = U
    return G


def _space_time_cross_meat(
    G1: np.ndarray, G2: np.ndarray, K: np.ndarray, time_lags: int = 0
) -> np.ndarray:
    """``sum_t G1[t]' K G2[t] + sum_{l=1..L} w_l (sum_t G1[t]' K G2[t-l]
    + sum_t G1[t-l]' K G2[t])`` with ``w_l = 1 - l/(L+1)``.

    With ``G1 = G2`` this is exactly :func:`spatial_hac_panel_meat` on the same
    grid (the two lag terms collapse to ``G + G'`` because ``K`` is symmetric);
    ``K = ones`` gives Driscoll-Kraay and ``K = kernel_matrix(D, cutoff)``
    gives Conley.
    """
    T = G1.shape[0]
    KG2 = np.einsum("ij,tjf->tif", K, G2)
    S = np.einsum("tif,tig->fg", G1, KG2)
    L = int(time_lags)
    for ell in range(1, L + 1):
        if ell >= T:
            break
        w = 1.0 - ell / (L + 1.0)
        A = np.einsum("tif,tig->fg", G1[ell:], KG2[:-ell])
        B = np.einsum("tif,tig->fg", G1[:-ell], KG2[ell:])
        S += w * (A + B)
    return S


# ==========================================================================
# Result
# ==========================================================================
class SpatialLPResult(LPResult):
    """Result of :func:`spatial_lp`.

    The object IS the per-horizon impulse-response table (an
    :class:`~puremacro.lp._results.LPResult`, hence a
    :class:`pandas.DataFrame`) in the package's multi-coefficient layout, so
    ``.point``, ``.se``, ``.ci_lower``, ``.ci_upper``, ``.t_stat``,
    ``.labels``, slicing and ``.plot()`` all work as they do for
    ``lp_state_dep`` / ``lp_asymmetric``.

    Columns
    -------
    ``h``, then ``beta_<label>``, ``se_<label>``, ``t_<label>``,
    ``lo_<label>``, ``hi_<label>`` for every label, then ``n_obs``,
    ``n_entities``, ``n_periods``.  The labels are, in column order:

    * ``direct`` — ``beta_h``;
    * ONE LABEL PER SPILLOVER ORDER, carrying ``c_p gamma_{p,h}``:
      ``indirect`` for order 1, then ``indirect2``, ``indirect3``, ... named
      by the ORDER (so ``spillover_orders=(1, 3)`` gives ``indirect`` and
      ``indirect3``).  Every per-order coefficient therefore reaches
      ``.point``, ``.ci_lower``, ``.labels``, slicing and ``.plot()``; the
      raw, UNSCALED ``gamma_{p,h}`` stays in :attr:`spillover`;
    * ``indirect_all`` — the aggregate ``sum_p c_p gamma_{p,h}``, present ONLY
      when there is more than one order (with a single order the per-order
      label already is the aggregate);
    * ``total`` — ``direct + indirect_all``.

    With the default ``spillover_orders=(1,)`` that is
    ``('direct', 'indirect', 'total')`` and 19 columns; with
    ``spillover_orders=(1, 2)`` it is
    ``('direct', 'indirect', 'indirect2', 'indirect_all', 'total')`` and 29.
    ``spillover_orders=()`` keeps an identically-zero ``indirect`` so the
    frame still has ``panel_lp``'s three labels.  :attr:`cumulative` carries
    ``cum_<label>``, ``se_<label>``, ``lo_<label>``, ``hi_<label>`` for the
    same labels.

    Attributes
    ----------
    spillover : pd.DataFrame
        Raw per-order coefficients behind ``indirect``: columns
        ``[h, order, gamma, se, t, lo, hi, row_sum_mean, scale, corr_own]``.
        ``corr_own`` is ``corrcoef(X_within[:, 0], X_within[:, 1 + i])`` — the
        correlation of the TWO-WAY DEMEANED own and spillover columns, which is
        the quantity that governs identification once the fixed effects are
        partialled out.
    cumulative : pd.DataFrame or None
        Running sums over the horizons with covariance-aware bands (columns
        ``[h]`` then ``cum_/se_/lo_/hi_`` for every frame label, then
        ``complete``); ``None`` when ``cumulative=False`` or the
        horizon grid is not contiguous from 0.  Standard errors come from
        ``r' C r`` on the cross-horizon covariance, never from summing
        per-horizon variances.
    coef : np.ndarray
        ``(H, F)`` focal point estimates in design order (``beta_h`` first,
        then ``gamma_{p,h}`` in ``spillover_orders`` order); NaN for empty
        horizons.
    coef_names : tuple of str
        Length ``F``; names the rows/columns of each ``vcov`` block.
    vcov : np.ndarray or None
        ``(H*F, H*F)`` cross-horizon covariance of the stacked focal
        coefficients, horizon-major (row ``m*F + f``).  Its DIAGONAL BLOCKS ARE
        the per-horizon sandwiches by construction, so ``sqrt(diag(vcov))``
        reproduces the reported standard errors exactly; the off-diagonal
        blocks are computed on the common (period x entity) grid with the
        single bandwidth in ``cov_options['cross_bandwidth']``.  ``None`` when
        ``cross_horizon=False``.
    horizon_values : np.ndarray
        Horizons, ascending: the row order of the frame and of ``coef``, and
        the block order of ``vcov``.
    total_scale : np.ndarray
        The length-``F`` contrast ``c = (1, c_1, ..., c_P)`` actually used to
        build ``indirect`` and ``total``.
    spillover_orders : tuple of int
    weights_info : dict
        ``{order: {'kind', 'n_units', 'n_islands', 'islands', 'row_standardized',
        'row_sum_mean', 'row_sum_min', 'row_sum_max'}}`` — one entry per order.
    cov_type : str
        Normalised: ``'cluster'``, ``'driscoll-kraay'`` or ``'conley'``.
    cov_options : dict
        ``{'cutoff_km', 'time_lags', 'kernel', 'metric', 'bandwidth_by_horizon',
        'cross_bandwidth'}``.
    alpha, n_lags, y_name, x_name : float / int / str
    n_entities_panel, n_periods_panel : int
        Panel-level counts (the per-horizon counts are the ``n_entities`` /
        ``n_periods`` COLUMNS).
    spec : dict
        ``{'lag_spillover', 'spatial_lag_y', 'spatial_lag_controls', 'controls',
        'unbalanced', 'zero_lower_orders', 'n_regressors', 'weak_id_corr',
        'cross_horizon'}``.
    notes : tuple of str
        Caveats raised during estimation (islands, non-unit row sums, weak
        identification, dropped rows); also printed by :meth:`summary`.
    """

    _metadata = LPResult._metadata + [
        "spillover", "cumulative", "coef", "coef_names", "horizon_values",
        "total_scale", "spillover_orders", "weights_info", "cov_type",
        "cov_options", "alpha", "n_lags", "n_entities_panel", "n_periods_panel",
        "spec", "notes",
    ]

    @property
    def _constructor(self):
        return SpatialLPResult

    @property
    def metadata(self) -> dict[str, Any]:
        """Estimation metadata dictionary (including the spatial settings)."""
        out = super().metadata
        for key in ("cov_type", "cov_options", "spillover_orders", "total_scale",
                    "n_lags", "alpha", "spec", "weights_info"):
            out[key] = getattr(self, key, None)
        return out

    # -- presentation ------------------------------------------------------
    def summary(self) -> str:
        """Text summary: the specification, the errors, the honest reading of
        ``total``, the per-horizon table and the caveats."""
        orders = tuple(getattr(self, "spillover_orders", ()) or ())
        spec = dict(getattr(self, "spec", {}) or {})
        info = dict(getattr(self, "weights_info", {}) or {})
        opts = dict(getattr(self, "cov_options", {}) or {})
        c = np.asarray(getattr(self, "total_scale", np.array([1.0])), dtype=float)

        lines = ["Spatial local projection (direct / indirect / direct+indirect)"]
        y_name = getattr(self, "y_name", None)
        x_name = getattr(self, "x_name", None)
        w1 = info.get(orders[0]) if orders else None
        wtxt = ""
        if w1 is not None:
            wtxt = (f"   W: {w1['kind']} ({w1['n_units']} units, {w1['n_islands']} islands, "
                    + ("row-standardised" if w1["row_standardized"] else "not row-standardised") + ")")
        lines.append(f"  y: {y_name}   x: {x_name}{wtxt}")
        lines.append(
            f"  spec: two-way FE, {getattr(self, 'n_lags', '?')} lags, spillover orders "
            f"{orders if orders else '()'}, lagged spillover "
            + ("on" if spec.get("lag_spillover") else "off")
            + f", {spec.get('n_regressors', '?')} regressors"
        )
        cov = getattr(self, "cov_type", "?")
        if cov == "conley":
            errs = (f"Conley space-time HAC (cutoff {opts.get('cutoff_km')} km, "
                    f"{opts.get('kernel')}, {opts.get('metric')})")
        elif cov == "driscoll-kraay":
            errs = "Driscoll-Kraay HAC"
        else:
            errs = "cluster-robust by entity"
        lines.append(
            f"  errors: {errs} — {getattr(self, 'n_entities_panel', '?')} entities x "
            f"{getattr(self, 'n_periods_panel', '?')} periods, no finite-sample correction"
        )
        if orders:
            terms = " + ".join(f"{c[1 + i]:.3f} x gamma_{p}" for i, p in enumerate(orders))
            agg = "indirect_all" if len(orders) > 1 else "indirect"
            lines.append(f"  total = direct + {agg}, {agg} = {terms}")
            if len(orders) > 1:
                per_order = ", ".join(
                    f"{_indirect_label(p)} = {c[1 + i]:.3f} x gamma_{p}"
                    for i, p in enumerate(orders))
                lines.append(f"  per order: {per_order}")
        lines.append(
            "  reading: beta_h and gamma_h are responses RELATIVE TO THE PERIOD MEAN "
            "(the two-way time effect absorbs the aggregate component), so `total` is a "
            "cross-sectional contrast, NOT the response to a uniform shock."
        )
        lines.append("")
        lines.append(LPResult.summary(self))
        cum = getattr(self, "cumulative", None)
        if cum is not None and len(cum):
            last = cum.iloc[-1]
            agg = "indirect_all" if "cum_indirect_all" in cum.columns else "indirect"
            parts = ", ".join(
                f"{lab} {last[f'cum_{lab}']:.3f} ({last[f'se_{lab}']:.3f})"
                for lab in ("direct", agg, "total") if f"cum_{lab}" in cum.columns)
            lines.append(
                f"  cumulative to h={int(last['h'])}: " + parts
                + ("" if bool(last["complete"]) else "  [incomplete: an empty horizon was skipped]")
            )
        lines.append(
            "  omitted: no joint cross-horizon spillover test ships — its measured size "
            "was 0.31 at a nominal 0.10, against 0.08-0.11 for the per-horizon z-test "
            "that does ship (see the module docstring)."
        )
        for note in getattr(self, "notes", ()) or ():
            lines.append(f"  note: {note}")
        return "\n".join(lines)

    def plot(
        self,
        *,
        kind: str = "panels",
        components: Sequence[str] | None = None,
        scale: float = 1.0,
        title: str = "",
        ax=None,
        figsize: tuple[float, float] = (10.5, 3.2),
    ):
        """Plot the direct / indirect / total responses.

        Parameters
        ----------
        kind : {'panels', 'cumulative', 'overlay'}
            ``'panels'`` draws one subplot per component from the per-horizon
            table; ``'cumulative'`` does the same from ``self.cumulative``;
            ``'overlay'`` puts the labelled series on one axis via
            :func:`puremacro.plot.plot_irf_single`.
        components : sequence of str, optional
            Any subset of :attr:`labels` (``'direct'``, one label per
            spillover order, ``'indirect_all'`` when there is more than one
            order, and ``'total'``).  ``None`` (default) draws them all.
        scale : float
            Multiplies every series and band.
        title : str
        ax : matplotlib Axes, optional
            Honoured only for a single component (``kind='panels'`` /
            ``'cumulative'``) or for ``kind='overlay'``.
        figsize : tuple

        Returns
        -------
        matplotlib.figure.Figure

        Raises
        ------
        ValueError
            For an unknown ``kind`` or component, when ``ax`` is given with
            more than one component, or when ``kind='cumulative'`` and no
            cumulative frame was built.
        """
        import matplotlib.pyplot as plt

        allowed = tuple(self.labels)
        comps = list(allowed) if components is None else [str(c) for c in components]
        bad = [c for c in comps if c not in allowed]
        if bad:
            raise ValueError(
                f"SpatialLPResult.plot: components must be a subset of {allowed}; got {bad}")
        if kind == "overlay":
            from ..plot import plot_irf_single
            keep = ["h"] + [f"{b}_{c}" for c in comps for b in ("beta", "se", "lo", "hi")]
            return plot_irf_single(pd.DataFrame(self)[keep], title=title, scale=scale, ax=ax)
        if kind == "panels":
            src = pd.DataFrame(self)
            prefix, label = "beta", "response"
        elif kind == "cumulative":
            src = getattr(self, "cumulative", None)
            if src is None:
                raise ValueError(
                    "SpatialLPResult.plot: kind='cumulative' needs the cumulative frame; "
                    "re-run spatial_lp with cumulative=True on a contiguous horizon grid")
            prefix, label = "cum", "cumulative response"
        else:
            raise ValueError(
                f"SpatialLPResult.plot: kind must be 'panels', 'cumulative' or 'overlay'; got {kind!r}")
        if ax is not None and len(comps) != 1:
            raise ValueError(
                "SpatialLPResult.plot: ax= is honoured only for a single component; "
                f"got components={tuple(comps)}")
        orders = tuple(getattr(self, "spillover_orders", ()) or ())
        titles = {"direct": "Direct (own shock)",
                  "indirect": ("Indirect (neighbour shock)" if len(orders) < 2
                               else "Indirect (order 1)"),
                  "indirect_all": "Indirect (all orders)",
                  "total": "Direct + indirect"}
        for p in orders:
            titles.setdefault(_indirect_label(p), f"Indirect (order {p})")
        if ax is not None:
            fig, axes = ax.get_figure(), [ax]
        else:
            fig, axes = plt.subplots(1, len(comps), figsize=figsize, sharex=True, squeeze=False)
            axes = list(axes[0])
        h = src["h"].to_numpy()
        for a, comp in zip(axes, comps):
            a.plot(h, src[f"{prefix}_{comp}"].to_numpy() * scale, color="0.0", linewidth=1.4)
            a.fill_between(h, src[f"lo_{comp}"].to_numpy() * scale,
                           src[f"hi_{comp}"].to_numpy() * scale,
                           color="0.75", alpha=0.5, linewidth=0)
            a.axhline(0.0, color="0.3", linewidth=0.6, linestyle=":")
            a.set_xlabel("Horizon (h)")
            a.set_title(titles[comp])
        axes[0].set_ylabel(label.capitalize())
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig


# ==========================================================================
# Argument resolution
# ==========================================================================
def _resolve_cov(
    cov_type: str,
    *,
    coords: Any,
    cutoff_km: float | None,
    time_lags: int | None,
    kernel: str | None,
    metric: str | None,
    func: str,
) -> tuple[str, CovFn, dict]:
    cov = str(cov_type).lower().replace("_", "-")
    if cov in _CLUSTER_ALIASES:
        kind = "cluster"
    elif cov in _DK_ALIASES:
        kind = "driscoll-kraay"
    elif cov in _CONLEY_ALIASES:
        kind = "conley"
    else:
        raise ValueError(
            f"{func}: cov_type must be 'cluster', 'driscoll-kraay' or 'conley'; "
            f"got {cov_type!r}")
    # G3: no silently-ignored keyword.
    if kind != "conley":
        for name, val in (("coords", coords), ("cutoff_km", cutoff_km),
                          ("kernel", kernel), ("metric", metric)):
            if val is not None:
                raise ValueError(
                    f"{func}: {name}= is meaningful only for cov_type='conley'; "
                    f"got cov_type={kind!r}. Drop {name}= or switch to cov_type='conley'.")
    if kind == "cluster" and time_lags is not None:
        raise ValueError(
            f"{func}: time_lags= is meaningful only for cov_type='driscoll-kraay' or "
            "'conley' (the cluster meat has no time kernel); got cov_type='cluster'.")
    if time_lags is not None and (isinstance(time_lags, bool)
                                  or not float(time_lags).is_integer()
                                  or int(time_lags) < 0):
        raise ValueError(f"{func}: time_lags must be a non-negative integer, got {time_lags!r}")

    opts = {"cutoff_km": None, "time_lags": None if time_lags is None else int(time_lags),
            "kernel": None, "metric": None, "_coords": None}
    if kind == "cluster":
        return kind, _cluster_cov, opts
    if kind == "driscoll-kraay":
        return kind, _make_dk_cov_fn(time_lags), opts
    if coords is None or cutoff_km is None:
        raise ValueError(
            f"{func}: cov_type='conley' needs coords= (one [lat, lon] per entity) and cutoff_km=")
    k = "bartlett" if kernel is None else str(kernel)
    m = "haversine" if metric is None else str(metric)
    opts["cutoff_km"] = float(cutoff_km)
    opts["kernel"] = k
    opts["metric"] = m
    opts["_coords"] = coords
    return kind, _make_conley_cov_fn(coords, cutoff_km=float(cutoff_km),
                                     time_lags=time_lags, kernel=k, metric=m), opts


def _resolve_total_scale(total_scale: Any, sbar: np.ndarray, func: str) -> np.ndarray:
    """Resolve ``total_scale`` ONCE into ``c = (1, c_1, ..., c_P)``."""
    P = len(sbar)
    if isinstance(total_scale, str):
        key = total_scale.lower()
        if key == "row_sum":
            c = np.asarray(sbar, dtype=float)
        elif key == "unit":
            c = np.ones(P, dtype=float)
        else:
            raise ValueError(
                f"{func}: total_scale must be 'row_sum', 'unit', a float, or one float "
                f"per spillover order ({P}), got {total_scale!r}")
    elif np.isscalar(total_scale):
        c = np.full(P, float(total_scale))
    else:
        arr = np.asarray(total_scale, dtype=float).ravel()
        if arr.shape[0] != P:
            raise ValueError(
                f"{func}: total_scale must be 'row_sum', 'unit', a float, or one float "
                f"per spillover order ({P}), got {total_scale!r}")
        c = arr
    return np.concatenate([[1.0], c])


def _indirect_label(order: int) -> str:
    """Frame label of the order-``p`` spillover: ``'indirect'`` for the first
    ring, ``'indirect2'``, ``'indirect3'``, ... above it."""
    return "indirect" if int(order) == 1 else f"indirect{int(order)}"


def _label_selectors(
    orders: Sequence[int], c_vec: np.ndarray, nF: int
) -> list[tuple[str, np.ndarray]]:
    """``[(label, contrast)]`` in frame-column order.

    ``direct`` (``beta_h``), then ONE ENTRY PER SPILLOVER ORDER
    (``c_p gamma_{p,h}``, labelled by :func:`_indirect_label`), then the
    aggregate ``indirect_all = sum_p c_p gamma_{p,h}`` when there is more than
    one order, then ``total = direct + indirect_all``.  With a single order the
    per-order entry IS the aggregate, so no ``indirect_all`` column is added
    and the layout is the documented 19-column one; with
    ``spillover_orders=()`` an identically-zero ``indirect`` is kept so that
    the frame still collapses onto ``panel_lp``'s layout.
    """
    direct = np.zeros(nF)
    direct[0] = 1.0
    agg = np.concatenate([[0.0], np.asarray(c_vec, dtype=float)[1:]])
    entries: list[tuple[str, np.ndarray]] = [("direct", direct)]
    if not tuple(orders):
        entries.append(("indirect", np.zeros(nF)))
    else:
        for i, p in enumerate(orders):
            sel = np.zeros(nF)
            sel[1 + i] = float(c_vec[1 + i])
            entries.append((_indirect_label(p), sel))
        if len(tuple(orders)) > 1:
            entries.append(("indirect_all", agg))
    entries.append(("total", direct + agg))
    return entries


# ==========================================================================
# The estimator
# ==========================================================================
def spatial_lp(
    df_wide: pd.DataFrame,
    y: str,
    x: str,
    weights: SpatialWeights,
    horizons: Iterable[int] = range(0, 21),
    n_lags: int = 2,
    controls: Sequence[str] = (),
    alpha: float = 0.10,
    entity_level: str = "code",
    time_level: str = "date",
    *,
    lags: int | None = None,
    horizon: int | None = None,
    ci: float | None = None,
    unit_col: str | None = None,
    time_col: str | None = None,
    spillover_orders: Sequence[int] = (1,),
    zero_lower_orders: bool = True,
    lag_spillover: bool = True,
    spatial_lag_y: bool = False,
    spatial_lag_controls: bool = False,
    total_scale: Any = "row_sum",
    unbalanced: str = "propagate",
    cumulative: bool = True,
    cross_horizon: bool = True,
    cov_type: str = "driscoll-kraay",
    coords: Any = None,
    cutoff_km: float | None = None,
    time_lags: int | None = None,
    kernel: str | None = None,
    metric: str | None = None,
) -> SpatialLPResult:
    """Two-way FE panel local projection augmented with the spatially lagged shock.

    Estimates, for every horizon ``h``,

    ``y_{i,t+h} - y_{i,t-1} = mu_i + tau_t + beta_h x_{i,t}
    + sum_p gamma_{p,h} (W^(p) x)_{i,t} + lags + eps``

    and reports ``direct = beta_h``, one ``c_p gamma_{p,h}`` column per
    spillover order (``indirect``, ``indirect2``, ...), their aggregate
    ``indirect_all`` when there is more than one order, and
    ``total = direct + sum_p c_p gamma_{p,h}`` — each with a covariance-aware
    standard error.  See :class:`SpatialLPResult` for the exact column layout.

    The signature mirrors :func:`puremacro.lp.panel_lp` — same first three
    positional arguments, same ``horizons`` / ``n_lags`` / ``controls`` /
    ``alpha`` / ``entity_level`` / ``time_level``, same ``lags`` / ``horizon``
    / ``ci`` / ``unit_col`` / ``time_col`` aliases, same ``cov_type`` block —
    with ``weights`` inserted as the fourth positional argument.  With
    ``spillover_orders=()`` the specification collapses onto ``panel_lp``
    bit for bit.  There is no ``**kwargs`` sink, so a typo raises ``TypeError``.

    Parameters
    ----------
    df_wide : pd.DataFrame
        Panel indexed by a ``(entity, time)`` MultiIndex (level names given by
        ``entity_level`` / ``time_level``), or a long-form frame whose entity
        and time identifiers are the columns ``unit_col`` / ``time_col``.
    y, x : str
        Outcome and shock column names.
    weights : SpatialWeights
        First-order spatial weights.  ``weights.ids`` must be exactly the
        panel's entity set.
    horizons : iterable of int, default ``range(0, 21)``
        Strictly increasing horizons.
    n_lags : int, default 2
        Own lags of ``y`` and ``x`` (and of the spatial lags, the controls and
        their spatial lags when the corresponding flag is on).
    controls : sequence of str, default ``()``
        Extra contemporaneous controls (their lags are added too, as in
        ``panel_lp``).
    alpha : float, default 0.10
        Two-sided level; bands are ``+- z_{1-alpha/2} * se`` (normal quantile,
        as in ``panel_lp``; no t correction).
    entity_level, time_level : str
        MultiIndex level names.
    lags, horizon, ci : optional
        The 2.0 aliases for ``n_lags`` / ``max(horizons)`` / ``1 - alpha``.
    unit_col, time_col : str, optional
        Long-form entity / time columns.
    spillover_orders : sequence of int, default ``(1,)``
        Neighbour orders to include.  Positive and distinct.  Each order gets
        its OWN frame label (``indirect`` for order 1, ``indirect2`` for order
        2, ...) plus an aggregate ``indirect_all`` when there is more than one.
        ``()`` estimates the plain ``panel_lp`` specification (and then
        ``indirect`` and ``total`` are identically zero and ``beta_direct``).
    zero_lower_orders : bool, default True
        Anselin (1988) higher-order convention (see
        :func:`higher_order_weights`); only relevant for orders ``>= 2``.
    lag_spillover : bool, default True
        Include ``n_lags`` lags of every ``W^(p) x`` column.
    spatial_lag_y : bool, default False
        Include ``n_lags`` LAGS of every ``W^(p) y`` column.  Off by default:
        with them the specification becomes a dynamic spatial panel whose
        two-way FE bias is ``O(1/T)`` and whose ``gamma`` is conditional on
        neighbour outcome history; without them ``gamma`` can absorb the
        reduced-form propagation of past neighbour outcomes.  Say which one you
        ran.  A CONTEMPORANEOUS ``W y`` is never added (it is endogenous).
    spatial_lag_controls : bool, default False
        Include contemporaneous and lagged ``W^(p) c`` for every control.
    total_scale : {'row_sum', 'unit'}, float or sequence of float, default 'row_sum'
        Resolved ONCE into ``c = (1, c_1, ..., c_P)`` and used everywhere
        (``indirect``, ``total``, their standard errors and the cumulative
        selector).  ``'row_sum'`` sets ``c_p`` to the mean row sum of
        ``W^(p)`` (1 for a row-standardised island-free ``W``); ``'unit'`` sets
        every ``c_p = 1``.
    unbalanced : {'propagate', 'renormalize'}, default 'propagate'
        Missing-neighbour policy of the spatial lag; see
        :func:`spatial_lag_panel`.
    cumulative : bool, default True
        Build the running-sum frame.  Requires a contiguous horizon grid
        starting at 0 and ``cross_horizon=True``.
    cross_horizon : bool, default True
        Build the ``(H*F, H*F)`` cross-horizon covariance.  Setting it to
        ``False`` saves ``H(H+1)/2`` block products and leaves ``.vcov`` and
        ``.cumulative`` as ``None``; the per-horizon standard errors are
        unaffected.
    cov_type : {'cluster', 'driscoll-kraay', 'conley'}, default 'driscoll-kraay'
        Same alias sets as ``panel_lp``.  The default DIFFERS from
        ``panel_lp``'s ``'cluster'`` deliberately: cluster-by-entity assumes
        independence across entities, which this regression denies by
        construction (``(W x)_i`` is a function of other units' shocks and the
        residuals are spatially correlated), so it is expected to under-cover
        and is offered only for comparability.
    coords : DataFrame or mapping, optional
        One ``[lat, lon]`` per entity; required by (and only accepted with)
        ``cov_type='conley'``.
    cutoff_km : float, optional
        Conley distance cutoff.  Required by (and only accepted with)
        ``'conley'``.
    time_lags : int, optional
        Time bandwidth for ``'driscoll-kraay'`` / ``'conley'``; ``None`` uses
        ``floor(4 (T/100)^(2/9))`` on the number of periods.  Rejected for
        ``'cluster'``.
    kernel : {'bartlett', 'uniform'}, optional
        Conley spatial kernel (default ``'bartlett'``).  Rejected for the
        other covariance types.
    metric : {'haversine', 'euclidean'}, optional
        Conley distance metric (default ``'haversine'``).  Rejected for the
        other covariance types.

    Returns
    -------
    SpatialLPResult
        The per-horizon table, with ``.spillover``, ``.cumulative``, ``.coef``,
        ``.vcov`` and the metadata attached.

    Raises
    ------
    ValueError
        ``horizons`` empty or not strictly increasing; ``spillover_orders``
        containing a non-positive, non-integer or duplicate order;
        ``spatial_lag_y`` / ``spatial_lag_controls`` with no spillover order;
        an unknown ``cov_type`` / ``unbalanced`` / ``total_scale``; a keyword
        that the chosen ``cov_type`` ignores; a generated column name colliding
        with a user column; a duplicated ``(entity, time)`` row; fewer than two
        entities or two periods; weights ids that are not panel entities; a
        spillover regressor that is identically zero (all units are islands or
        ``W`` is the zero matrix); ``cumulative=True`` on a non-contiguous
        horizon grid or with ``cross_horizon=False``.
    KeyError
        Panel entities missing from ``weights`` (or, for Conley, from
        ``coords``).
    numpy.linalg.LinAlgError
        A singular within design; the message names the offending column
        LABELS (``puremacro._linalg.inv_xtx`` reports positional indices,
        which are translated here).
    TypeError
        Any unknown keyword — there is no ``**kwargs`` sink.

    Warns
    -----
    RuntimeWarning
        Islands at some order; ``W`` row sums not all equal; within-sample
        ``|corr(x, W x)| > 0.999`` at some horizon (``gamma`` weakly
        identified); rows dropped because a neighbour's shock was missing under
        ``unbalanced='propagate'``; a negative cumulative variance.

    Notes
    -----
    **Identification.**  ``beta_h`` and ``gamma_{p,h}`` are relative to the
    period mean; ``total`` is a cross-sectional contrast, not an aggregate
    multiplier (see the module docstring).  ``gamma`` is identified purely off
    the cross-sectional variation in ``W x`` given ``x``, so a near-uniform
    shock, a dense ``W``, or a ``W`` chosen after looking at the results all
    produce weakly identified or specification-searched estimates.  Report more
    than one ``W`` (contiguity, knn, economic) side by side.

    **Exogeneity.**  ``x`` enters as an exogenous regressor conditional on the
    two-way fixed effects and the included lags.  No IV path ships.

    **Asymptotics.**  Cluster-by-entity needs independence across entities,
    which this regression denies; Driscoll-Kraay is robust to arbitrary
    cross-sectional dependence and to the MA(h) serial correlation the
    overlapping horizons induce, and is valid as ``T -> infinity``; Conley is
    valid as the number of units grows with correlation decaying inside
    ``cutoff_km``, and is the recommended choice for the ``T ~ 40-100``
    regional panels this module targets.

    **Weights and the panel.**  A weights id that never appears in the panel is
    an error, not something ``unbalanced='renormalize'`` repairs: a unit absent
    from the whole sample changes the POPULATION the weights describe (who is
    an island, what every row sum means), and the fix is to rebuild ``W`` on
    the estimation sample.  ``renormalize`` handles per-period holes, which
    leave the population intact.

    **Scales.**  ``c_p`` under ``total_scale='row_sum'`` is the mean row sum of
    ``W^(p)`` over ALL units of ``W``, fixed once, not the mean over the units
    that actually contribute at horizon ``h``.  When the estimation sample
    shrinks with ``h`` the two differ; ``weights_info`` and the per-horizon
    ``n_entities`` column let you check by how much, and ``total_scale=`` takes
    an explicit value.

    **Empty horizons.**  A horizon whose sample is empty, or which the two-way
    within transformation annihilates (fewer than two surviving entities or
    periods), reports NaN estimates with ``n_obs = n_entities = n_periods = 0``
    and a ``RuntimeWarning``; it is skipped by the cumulative selector and every
    cumulative row from there on carries ``complete = False``.

    **Bandwidths.**  Each horizon's reported standard error uses that horizon's
    own sample and its own bandwidth, exactly as ``panel_lp`` does (recorded in
    ``cov_options['bandwidth_by_horizon']``); the OFF-DIAGONAL blocks of the
    cross-horizon covariance use one bandwidth for the whole object
    (``cov_options['cross_bandwidth']``, from the union period axis).  The
    diagonal blocks are the per-horizon sandwiches themselves, so
    ``sqrt(diag(vcov))`` matches the reported standard errors by construction;
    the price is that the assembled matrix is not guaranteed positive
    semi-definite when the horizon samples are not nested.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> from puremacro.spatial import knn_weights
    >>> from puremacro.spatial.lp import spatial_lp
    >>> rng = np.random.default_rng(0)
    >>> ids = [f"R{i:02d}" for i in range(12)]
    >>> coords = pd.DataFrame({"lat": rng.uniform(30, 45, 12),
    ...                        "lon": rng.uniform(-110, -75, 12)}, index=ids)
    >>> W = knn_weights(coords, 3)
    >>> T = 80
    >>> x = rng.standard_normal((12, T))
    >>> dy = 0.5 * x + 0.3 * (W.W @ x) + 0.2 * rng.standard_normal((12, T))
    >>> yv = np.cumsum(dy, axis=1)
    >>> df = pd.DataFrame({"code": np.repeat(ids, T),
    ...                    "date": np.tile(np.arange(T), 12),
    ...                    "x": x.ravel(), "y": yv.ravel()}).set_index(["code", "date"])
    >>> res = spatial_lp(df, "y", "x", W, horizons=range(0, 4), n_lags=1)
    >>> list(res.labels)
    ['direct', 'indirect', 'total']
    >>> bool(abs(res["beta_direct"].iloc[0] - 0.5) < 0.05)
    True
    >>> bool(abs(res["beta_indirect"].iloc[0] - 0.3) < 0.1)
    True
    >>> res.vcov.shape == (4 * 2, 4 * 2)
    True
    """
    func = "spatial_lp"
    horizons, n_lags, alpha = resolve_lp_kwargs(
        horizons, n_lags, alpha, lags=lags, horizon=horizon, ci=ci, name=func)
    if len(horizons) == 0:
        raise ValueError(f"{func}: horizons is empty; pass at least one horizon")
    if any(b <= a for a, b in zip(horizons[:-1], horizons[1:])):
        raise ValueError(
            f"{func}: horizons must be strictly increasing (the cumulative frame is a "
            f"running sum over them); got {horizons}")

    orders_raw = tuple(spillover_orders)
    for p in orders_raw:
        if isinstance(p, bool) or not float(p).is_integer() or int(p) < 1:
            raise ValueError(
                f"{func}: spillover_orders must be positive distinct integers; got "
                f"{orders_raw} — order 0 is the own shock, which is already the first "
                "regressor")
    orders = tuple(int(p) for p in orders_raw)
    if len(set(orders)) != len(orders):
        raise ValueError(
            f"{func}: spillover_orders must be positive distinct integers; got {orders_raw}")
    if not orders:
        for name, flag in (("spatial_lag_y", spatial_lag_y),
                           ("spatial_lag_controls", spatial_lag_controls)):
            if flag:
                raise ValueError(
                    f"{func}: {name}=True requires at least one spillover order; got "
                    "spillover_orders=()")
    if unbalanced not in ("propagate", "renormalize"):
        raise ValueError(
            f"{func}: unbalanced must be 'propagate' or 'renormalize', got {unbalanced!r}")

    cov_kind, cov_fn, cov_options = _resolve_cov(
        cov_type, coords=coords, cutoff_km=cutoff_km, time_lags=time_lags,
        kernel=kernel, metric=metric, func=func)

    contiguous = list(horizons) == list(range(0, len(horizons)))
    if cumulative and not contiguous:
        raise ValueError(
            f"{func}: cumulative=True needs a contiguous horizon grid starting at 0 "
            f"(the running sum and the Ramey-Zubairy multiplier numerator are only "
            f"defined there); got horizons={horizons}. Pass cumulative=False.")
    if cumulative and not cross_horizon:
        raise ValueError(
            f"{func}: cumulative=True needs cross_horizon=True (the bands come from "
            "r' C r on the cross-horizon covariance, never from summing per-horizon "
            "variances)")

    df, entity_level, time_level = as_panel_index(
        df_wide, entity_level=entity_level, time_level=time_level,
        unit_col=unit_col, time_col=time_col, func=func)
    df = df.copy()
    df.index = df.index.set_names([entity_level, time_level])
    df = df.sort_index()

    dup = int(df.index.duplicated().sum())
    if dup:
        examples = list(df.index[df.index.duplicated(keep=False)][:3])
        raise ValueError(
            f"{func}: the (entity, time) index has {dup} duplicated row(s), e.g. "
            f"{examples}; aggregate or de-duplicate the panel first")

    ctl = list(controls)
    for col in [y, x] + ctl:
        if col not in df.columns:
            raise KeyError(f"{func}: column {col!r} is not in the panel; columns are "
                           f"{list(df.columns)}")

    entities_panel = pd.Index(df.index.get_level_values(entity_level)).unique()
    periods_panel = pd.Index(df.index.get_level_values(time_level)).unique()
    short = []
    if len(periods_panel) < 2:
        short.append(f"{len(periods_panel)} period(s) (2 needed)")
    if len(entities_panel) < 2:
        short.append(f"{len(entities_panel)} entit(ies) (2 needed)")
    if short:
        raise ValueError(
            f"{func}: the panel has " + " and ".join(short) + " — the two-way within "
            "transformation annihilates anything smaller")
    ent_set = set(entities_panel)
    id_set = set(weights.ids)
    missing_ids = [u for u in entities_panel if u not in id_set]
    if missing_ids:
        raise KeyError(
            f"{func}: weights are missing {len(missing_ids)} panel entit(ies), e.g. "
            f"{missing_ids[:3]}")
    extra_ids = [u for u in weights.ids if u not in ent_set]
    if extra_ids:
        raise ValueError(
            f"{func}: weights carry {len(extra_ids)} id(s) that are not entities of the "
            f"panel, e.g. {extra_ids[:3]}; rebuild the weights on the estimation sample "
            "(subsetting W here would silently change its row sums, and with them every "
            "gamma's interpretation)")

    # ---------------------------------------------------------------- STEP 1
    Wp_by_order: dict[int, SpatialWeights] = {}
    weights_info: dict[int, dict] = {}
    notes: list[str] = []
    sbar = np.ones(len(orders))
    for i, p in enumerate(orders):
        Wp = weights if p == 1 else higher_order_weights(
            weights, p, zero_lower_orders=zero_lower_orders)
        Wp_by_order[p] = Wp
        rs = Wp.row_sums()
        sbar[i] = float(rs.mean())
        weights_info[p] = {
            "kind": Wp.kind, "n_units": Wp.n, "n_islands": Wp.n_islands,
            "islands": Wp.islands, "row_standardized": Wp.row_standardized,
            "row_sum_mean": float(rs.mean()), "row_sum_min": float(rs.min()),
            "row_sum_max": float(rs.max()),
        }
        if Wp.n_islands:
            note = (f"order {p}: {Wp.n_islands} island(s) {list(Wp.islands)[:3]} — their "
                    f"spillover regressor is 0 at every t, the mean row sum falls to "
                    f"{rs.mean():.3f}, and the effective population responding at that "
                    "order is smaller than n")
            notes.append(note)
            warnings.warn(f"{func}: {note}", RuntimeWarning, stacklevel=2)
        live_rs = rs[rs > 0]
        if live_rs.size and float(np.std(live_rs)) > ROW_SUM_TOL:
            note = (f"order {p}: the row sums of W are not all equal (min "
                    f"{live_rs.min():.3f}, max {live_rs.max():.3f} over the non-island "
                    f"rows), so gamma_{p} is the response to a one-unit rise "
                    "in the WEIGHTED SUM of neighbour shocks, not their average; "
                    f"total_scale='row_sum' uses the cross-sectional mean {rs.mean():.3f}")
            notes.append(note)
            warnings.warn(f"{func}: {note}", RuntimeWarning, stacklevel=2)

    c_vec = _resolve_total_scale(total_scale, sbar, func)

    # ---------------------------------------------------------------- STEP 2
    lag_targets: list[str] = [x]
    if spatial_lag_y:
        lag_targets.append(y)
    if spatial_lag_controls:
        lag_targets.extend(ctl)

    generated: list[str] = []
    for p in orders:
        for col in lag_targets:
            generated.append(f"__W{p}_{col}__")
    for lag in range(1, n_lags + 1):
        generated += [f"__{x}_L{lag}__", f"__{y}_L{lag}__"]
        generated += [f"__{cc}_L{lag}__" for cc in ctl]
        for p in orders:
            if lag_spillover:
                generated.append(f"__W{p}_{x}_L{lag}__")
            if spatial_lag_y:
                generated.append(f"__W{p}_{y}_L{lag}__")
            if spatial_lag_controls:
                generated += [f"__W{p}_{cc}_L{lag}__" for cc in ctl]
    generated += [f"__dy_h{h}__" for h in horizons]
    clash = sorted(set(generated) & set(map(str, df.columns)))
    if clash:
        raise ValueError(
            f"{func}: the generated column name(s) {clash[:3]} collide with columns of "
            "the input frame; rename them before calling (the estimator refuses to "
            "overwrite user data)")

    n_poisoned = 0
    for p in orders:
        lagged = spatial_lag_panel(
            df, lag_targets, Wp_by_order[p], entity_level=entity_level,
            time_level=time_level, prefix=f"__W{p}_", suffix="__",
            unbalanced=unbalanced)
        for col in lag_targets:
            name = f"__W{p}_{col}__"
            df[name] = lagged[name]
        wx = df[f"__W{p}_{x}__"].to_numpy(dtype=float)
        base = df[x].to_numpy(dtype=float)
        finite = np.isfinite(wx)
        if finite.any() and np.nanmax(np.abs(wx)) <= ZERO_COLUMN_TOL * (
                1.0 + np.nanmax(np.abs(base))):
            raise ValueError(
                f"{func}: the order-{p} spatial lag of {x!r} is identically zero "
                f"({Wp_by_order[p].n_islands} of {Wp_by_order[p].n} units are islands / W "
                "has no non-zero row), so the spillover regressor carries no variation. "
                "Use spillover_orders=() — which reproduces puremacro.lp.panel_lp "
                "exactly — or supply weights with neighbours.")
        n_poisoned += int(np.sum(np.isfinite(base) & ~finite))
    if n_poisoned and unbalanced == "propagate":
        note = (f"{n_poisoned} row(s) have an observed shock but an incomplete "
                "neighbourhood, so their spatial lag is NaN and they drop at every "
                "horizon; unbalanced='renormalize' rescales over the observed neighbours "
                "instead")
        notes.append(note)
        warnings.warn(f"{func}: {note}", RuntimeWarning, stacklevel=2)

    # ---------------------------------------------------------------- STEP 3
    grouped = df.groupby(level=entity_level)
    for lag in range(1, n_lags + 1):
        df[f"__{x}_L{lag}__"] = grouped[x].shift(lag)
        df[f"__{y}_L{lag}__"] = grouped[y].shift(lag)
        for cc in ctl:
            df[f"__{cc}_L{lag}__"] = grouped[cc].shift(lag)
        for p in orders:
            if lag_spillover:
                df[f"__W{p}_{x}_L{lag}__"] = grouped[f"__W{p}_{x}__"].shift(lag)
            if spatial_lag_y:
                df[f"__W{p}_{y}_L{lag}__"] = grouped[f"__W{p}_{y}__"].shift(lag)
            if spatial_lag_controls:
                for cc in ctl:
                    df[f"__W{p}_{cc}_L{lag}__"] = grouped[f"__W{p}_{cc}__"].shift(lag)

    x_cols = [x] + [f"__W{p}_{x}__" for p in orders]
    for lag in range(1, n_lags + 1):
        x_cols += [f"__{x}_L{lag}__", f"__{y}_L{lag}__"]
        if lag_spillover:
            x_cols += [f"__W{p}_{x}_L{lag}__" for p in orders]
        if spatial_lag_y:
            x_cols += [f"__W{p}_{y}_L{lag}__" for p in orders]
        x_cols += [f"__{cc}_L{lag}__" for cc in ctl]
        if spatial_lag_controls:
            x_cols += [f"__W{p}_{cc}_L{lag}__" for p in orders for cc in ctl]
    x_cols += list(ctl)
    if spatial_lag_controls:
        x_cols += [f"__W{p}_{cc}__" for p in orders for cc in ctl]

    F = list(range(1 + len(orders)))
    coef_names = tuple([x] + [f"W{p}_{x}" for p in orders])
    nF = len(F)
    z_crit = norm.ppf(1 - alpha / 2)
    label_sel = _label_selectors(orders, c_vec, nF)
    labels = [lab for lab, _ in label_sel]

    # ---------------------------------------------------------------- STEP 4
    H = len(horizons)
    coef = np.full((H, nF), np.nan)
    blocks: list[np.ndarray | None] = [None] * H
    psis: list[np.ndarray | None] = [None] * H
    ent_keys: list[Any] = [None] * H
    tim_keys: list[Any] = [None] * H
    bandwidths: dict[int, int | None] = {}
    corr_own = np.full((H, len(orders)), np.nan)
    rows: list[dict] = []
    spill_rows: list[dict] = []
    weak: list[int] = []

    for m, h in enumerate(horizons):
        col_dy = f"__dy_h{h}__"
        df[col_dy] = grouped[y].shift(-h) - grouped[y].shift(1)
        sub = df[[col_dy] + x_cols].dropna()
        n_per_h = sub.index.get_level_values(time_level).nunique() if len(sub) else 0
        n_ent_h = sub.index.get_level_values(entity_level).nunique() if len(sub) else 0
        if sub.empty or n_per_h < 2 or n_ent_h < 2:
            if len(sub):
                note = (f"horizon {h} keeps {n_ent_h} entit(ies) x {n_per_h} period(s); "
                        "the two-way within transformation annihilates it, so the row is "
                        "NaN and every cumulative row from there on is flagged incomplete")
            else:
                note = (f"horizon {h} has no observations at all after the lag/lead "
                        f"construction ({n_lags} lag(s) and a lead of {h} on "
                        f"{len(periods_panel)} period(s)), so the row is NaN and every "
                        "cumulative row from there on is flagged incomplete")
            notes.append(note)
            warnings.warn(f"{func}: {note}", RuntimeWarning, stacklevel=2)
            bandwidths[h] = None
            rows.append(_nan_row(h, labels))
            for p in orders:
                spill_rows.append({
                    "h": h, "order": p, "gamma": np.nan, "se": np.nan, "t": np.nan,
                    "lo": np.nan, "hi": np.nan,
                    "row_sum_mean": weights_info[p]["row_sum_mean"],
                    "scale": float(c_vec[1 + orders.index(p)]), "corr_own": np.nan,
                })
            continue
        try:
            out = two_way_fe_within(
                sub, y_col=col_dy, x_cols=x_cols,
                entity_level=entity_level, time_level=time_level)
        except np.linalg.LinAlgError as exc:
            raise np.linalg.LinAlgError(
                f"{func}: singular within design at horizon {h}. {exc} "
                f"Column labels in design order: {x_cols}."
            ) from exc

        V = cov_fn(out)
        blocks[m] = np.asarray(V)[np.ix_(F, F)]
        coef[m] = out["beta"][F]
        psis[m] = _focal_influence(out, F)
        ent_keys[m] = out["entity_keys"]
        tim_keys[m] = out["time_keys"]
        bandwidths[h] = (None if cov_kind == "cluster"
                         else _dk_bandwidth(out["time_keys"], cov_options["time_lags"]))

        Xw = out["X_within"]
        for i, p in enumerate(orders):
            with np.errstate(invalid="ignore", divide="ignore"):
                cc_ = np.corrcoef(Xw[:, 0], Xw[:, 1 + i])[0, 1]
            corr_own[m, i] = cc_
            if np.isfinite(cc_) and abs(cc_) > WEAK_ID_CORR:
                weak.append(h)

        Vb = blocks[m]
        row = {"h": h}
        for label, sel in label_sel:
            pt = float(sel @ coef[m])
            var = float(sel @ Vb @ sel)
            se = float(np.sqrt(var)) if var >= 0 else np.nan
            row[f"beta_{label}"] = pt
            row[f"se_{label}"] = se
            row[f"t_{label}"] = pt / se if (np.isfinite(se) and se > 0) else np.nan
            row[f"lo_{label}"] = pt - z_crit * se
            row[f"hi_{label}"] = pt + z_crit * se
        row["n_obs"] = int(out["n_obs"])
        row["n_entities"] = int(len(np.unique(out["entity_keys"])))
        row["n_periods"] = int(len(np.unique(out["time_keys"])))
        rows.append(row)

        for i, p in enumerate(orders):
            g = float(coef[m, 1 + i])
            sg = float(np.sqrt(Vb[1 + i, 1 + i])) if Vb[1 + i, 1 + i] >= 0 else np.nan
            spill_rows.append({
                "h": h, "order": p, "gamma": g, "se": sg,
                "t": g / sg if (np.isfinite(sg) and sg > 0) else np.nan,
                "lo": g - z_crit * sg, "hi": g + z_crit * sg,
                "row_sum_mean": weights_info[p]["row_sum_mean"],
                "scale": float(c_vec[1 + i]), "corr_own": float(corr_own[m, i]),
            })

    if weak:
        note = (f"the two-way demeaned own and spillover columns have |corr| > "
                f"{WEAK_ID_CORR} at horizon(s) {sorted(set(weak))[:5]}; gamma is weakly "
                "identified there (see spillover['corr_own'])")
        notes.append(note)
        warnings.warn(f"{func}: {note}", RuntimeWarning, stacklevel=2)

    # ---------------------------------------------------------------- STEP 5
    C: np.ndarray | None = None
    cross_bw: int | None = None
    if cross_horizon:
        C, cross_bw = _cross_horizon_cov(
            blocks, psis, ent_keys, tim_keys, cov_kind, cov_options, nF)
    cov_options = dict(cov_options)
    cov_options.pop("_coords", None)
    cov_options["bandwidth_by_horizon"] = bandwidths
    cov_options["cross_bandwidth"] = cross_bw

    # ---------------------------------------------------------------- STEP 7
    cum_frame = None
    if cumulative and C is not None:
        cum_frame = _cumulative_frame(horizons, coef, C, label_sel, nF, z_crit, func)

    # ---------------------------------------------------------------- result
    res = SpatialLPResult(rows)
    if "h" in res.columns:
        res.index = pd.Index(res["h"], name="h")
    res.y_name = str(y)
    res.x_name = str(x)
    res.method = "spatial_lp"
    res.ci_level = 1.0 - alpha
    res.alpha = float(alpha)
    res.n_lags = int(n_lags)
    res.spillover = pd.DataFrame(spill_rows)
    res.cumulative = cum_frame
    res.coef = coef
    res.coef_names = coef_names
    res.horizon_values = np.asarray(horizons, dtype=int)
    res.vcov = C
    res.total_scale = c_vec
    res.spillover_orders = orders
    res.weights_info = weights_info
    res.cov_type = cov_kind
    res.cov_options = cov_options
    res.n_entities_panel = int(len(entities_panel))
    res.n_periods_panel = int(len(periods_panel))
    res.spec = {
        "lag_spillover": bool(lag_spillover),
        "spatial_lag_y": bool(spatial_lag_y),
        "spatial_lag_controls": bool(spatial_lag_controls),
        "controls": tuple(ctl),
        "unbalanced": unbalanced,
        "zero_lower_orders": bool(zero_lower_orders),
        "n_regressors": len(x_cols),
        "weak_id_corr": WEAK_ID_CORR,
        "cross_horizon": bool(cross_horizon),
        "x_cols": tuple(x_cols),
    }
    res.notes = tuple(notes)
    return res


def _nan_row(h: int, labels: Sequence[str]) -> dict:
    row: dict = {"h": h}
    for label in labels:
        for base in ("beta", "se", "t", "lo", "hi"):
            row[f"{base}_{label}"] = np.nan
    row["n_obs"] = 0
    row["n_entities"] = 0
    row["n_periods"] = 0
    return row


def _cross_horizon_cov(
    blocks: Sequence[np.ndarray | None],
    psis: Sequence[np.ndarray | None],
    ent_keys: Sequence[Any],
    tim_keys: Sequence[Any],
    cov_kind: str,
    cov_options: dict,
    nF: int,
) -> tuple[np.ndarray, int | None]:
    """Assemble the ``(H*nF, H*nF)`` cross-horizon covariance.

    Diagonal blocks ARE the per-horizon sandwiches (so the reported standard
    errors are reproduced exactly); off-diagonal blocks come from the
    focal-influence identity on the common ``(period x entity)`` grid with a
    single bandwidth.
    """
    H = len(blocks)
    C = np.full((H * nF, H * nF), np.nan)
    live = [m for m in range(H) if blocks[m] is not None]
    for m in live:
        C[m * nF:(m + 1) * nF, m * nF:(m + 1) * nF] = blocks[m]
    if not live:
        return C, None

    entities = pd.Index(np.unique(np.concatenate(
        [np.asarray(ent_keys[m]) for m in live])))
    periods = pd.Index(np.unique(np.concatenate(
        [np.asarray(tim_keys[m]) for m in live])))
    grids = {m: _panel_score_grid(psis[m], ent_keys[m], tim_keys[m], entities, periods)
             for m in live}

    if cov_kind == "cluster":
        sums = {m: grids[m].sum(axis=0) for m in live}     # (n, nF)
        for ii, m in enumerate(live):
            for m2 in live[ii + 1:]:
                B = sums[m].T @ sums[m2]
                C[m * nF:(m + 1) * nF, m2 * nF:(m2 + 1) * nF] = B
                C[m2 * nF:(m2 + 1) * nF, m * nF:(m + 1) * nF] = B.T
        return C, None

    L = cov_options["time_lags"]
    if L is None:
        L = max(1, int(np.floor(4 * (len(periods) / 100) ** (2 / 9))))
    L = int(L)
    n_units = len(entities)
    if cov_kind == "driscoll-kraay":
        K = np.ones((n_units, n_units))
    else:
        Ccoords = _entity_coords(cov_options["_coords"], np.asarray(entities))
        K = kernel_matrix(pairwise_distances(Ccoords, cov_options["metric"]),
                          float(cov_options["cutoff_km"]), cov_options["kernel"])
    for ii, m in enumerate(live):
        for m2 in live[ii + 1:]:
            B = _space_time_cross_meat(grids[m], grids[m2], K, L)
            C[m * nF:(m + 1) * nF, m2 * nF:(m2 + 1) * nF] = B
            C[m2 * nF:(m2 + 1) * nF, m * nF:(m + 1) * nF] = B.T
    return C, L


def _cumulative_frame(
    horizons: Sequence[int],
    coef: np.ndarray,
    C: np.ndarray,
    label_sel: Sequence[tuple[str, np.ndarray]],
    nF: int,
    z_crit: float,
    func: str,
) -> pd.DataFrame:
    """Running sums with ``sqrt(r' C r)`` bands, one block per frame label."""
    H = len(horizons)
    theta = coef.ravel()
    live = np.isfinite(theta)
    sel = {label: np.asarray(s, dtype=float) for label, s in label_sel}
    rows = []
    complete = True
    negvar = False
    for m in range(H):
        if not np.isfinite(coef[m]).all():
            complete = False
        row = {"h": int(horizons[m])}
        for label, s in sel.items():
            r = np.zeros(H * nF)
            for j in range(m + 1):
                if np.isfinite(coef[j]).all():
                    r[j * nF:(j + 1) * nF] = s
            keep = live & (r != 0)
            pt = float(r[keep] @ theta[keep]) if keep.any() else 0.0
            sub = C[np.ix_(np.flatnonzero(keep), np.flatnonzero(keep))]
            var = float(r[keep] @ sub @ r[keep]) if keep.any() else 0.0
            if var < 0:
                negvar = True
                se = np.nan
            else:
                se = float(np.sqrt(var))
            row[f"cum_{label}"] = pt
            row[f"se_{label}"] = se
            row[f"lo_{label}"] = pt - z_crit * se
            row[f"hi_{label}"] = pt + z_crit * se
        row["complete"] = complete
        rows.append(row)
    if negvar:
        warnings.warn(
            f"{func}: a cumulative variance came out negative and its standard error is "
            "NaN. The cross-horizon covariance mixes each horizon's own sandwich on the "
            "diagonal with common-grid off-diagonal blocks, so it is not guaranteed "
            "positive semi-definite when the horizon samples are not nested; try "
            "kernel='uniform', a smaller cutoff, or an explicit time_lags.",
            RuntimeWarning, stacklevel=3,
        )
    cols = (["h"] + [f"{b}_{lab}" for lab in sel
                     for b in ("cum", "se", "lo", "hi")] + ["complete"])
    return pd.DataFrame(rows)[cols]
