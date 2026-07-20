"""de Chaisemartin-D'Haultfoeuille (2020) DID_M / DID_M^l estimator.

When treatment effects are heterogeneous across units and over time,
the standard two-way fixed-effects estimator can place **negative
weights** on some treatment effects — a contaminating problem
identified in CdH (2020). The DID_M estimator avoids this by
restricting attention to *switchers*: units that change treatment
status between two consecutive periods. For each switching event at
time ``t``, the change in outcome among switchers is compared to the
contemporaneous change among **stable** units (units whose treatment
status was the same at ``t - 1`` and ``t``).

Formally, in the simplest 0 -> 1 absorbing-treatment case, the
instantaneous estimator at calendar time ``t`` is::

    DID_M_t  =  E[Y_t - Y_{t-1} | switcher at t]
                  - E[Y_t - Y_{t-1} | D_{t-1} = D_t = 0]

DID_M then averages over ``t`` weighted by the number of switchers.
Long-run effects extend the comparison window to horizon ``l``::

    DID_M_t^l  =  E[Y_{t+l-1} - Y_{t-1} | switcher at t]
                    - E[Y_{t+l-1} - Y_{t-1} | stable from t-1 through t+l-1]

The "switchers placebo" reverses time and tests for a pre-event
violation of parallel trends::

    DID_M_t^placebo  =  E[Y_{t-1} - Y_{t-2} | switcher at t]
                          - E[Y_{t-1} - Y_{t-2} | stable]

Under the null (no anticipation, parallel trends) the placebo p-value
is uniform on [0, 1]. We report a single uniform p-value combining all
calendar times via a unit-resampling bootstrap.

Standard errors are computed by **cluster bootstrap** (resampling
panel units with replacement).

References
----------
de Chaisemartin, C. and D'Haultfoeuille, X. (2020). Two-way fixed
    effects estimators with heterogeneous treatment effects. American
    Economic Review 110(9), 2964-2996.
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd

from ._results import CdHResult


# ---------------------------------------------------------------------------
# Core point-estimate machinery (operates on a contiguous balanced panel)
# ---------------------------------------------------------------------------


def _build_panel_arrays(
    y: np.ndarray, d: np.ndarray, p: np.ndarray, t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pivot long arrays into wide ``(n_units, n_times)`` matrices.

    Returns
    -------
    Y, D, units, times
        ``Y``, ``D``: shape ``(n_units, n_times)`` with ``NaN`` where
        a (unit, time) pair is missing from the input. ``units``,
        ``times`` are the row / column labels in sort order.
    """
    units = np.unique(p)
    times = np.unique(t)
    u_idx = {u: i for i, u in enumerate(units)}
    t_idx = {tt: j for j, tt in enumerate(times)}
    Y = np.full((len(units), len(times)), np.nan)
    D = np.full((len(units), len(times)), np.nan)
    for yi, di, pi, ti in zip(y, d, p, t):
        i = u_idx[pi]; j = t_idx[ti]
        Y[i, j] = yi
        D[i, j] = di
    return Y, D, units, times


def _cdh_point(
    Y: np.ndarray,
    D: np.ndarray,
    *,
    horizons: tuple,
    placebo: bool,
):
    """Compute DID_M, DID_M^l, and (optionally) the switchers-placebo
    DID_M^placebo on a wide ``(n_units, n_times)`` panel.

    Returns
    -------
    att_M, att_M_l, att_M_placebo, n_switchers
        Scalars / 1-D arrays. ``att_M_placebo`` is ``np.nan`` if
        ``placebo=False`` or no placebo events are observed.
    """
    n_units, T = Y.shape
    L = len(horizons)

    # We accumulate (numerator, denominator) for each estimand so the
    # final value is a switcher-count-weighted mean across calendar
    # times and units.
    num_M = 0.0
    den_M = 0
    num_Ml = np.zeros(L)
    den_Ml = np.zeros(L, dtype=int)
    num_pl = 0.0
    den_pl = 0
    n_switchers = 0

    for tt in range(1, T):
        # Switchers 0 -> 1 from tt-1 to tt.
        d_prev = D[:, tt - 1]
        d_now = D[:, tt]
        valid_pair = (~np.isnan(d_prev)) & (~np.isnan(d_now))
        switcher = valid_pair & (d_prev == 0) & (d_now == 1)
        stable = valid_pair & (d_prev == 0) & (d_now == 0)

        # Instantaneous DID_M.
        if switcher.any() and stable.any():
            dy_sw = Y[switcher, tt] - Y[switcher, tt - 1]
            dy_st = Y[stable, tt] - Y[stable, tt - 1]
            mask_sw = ~np.isnan(dy_sw)
            mask_st = ~np.isnan(dy_st)
            if mask_sw.any() and mask_st.any():
                contribution = (dy_sw[mask_sw].mean()
                                 - dy_st[mask_st].mean())
                # Weight by number of switchers at this t.
                w = int(mask_sw.sum())
                num_M += contribution * w
                den_M += w
                n_switchers += w

        # Long-run DID_M^l. For horizon l (l = 1 means same as
        # instantaneous when l == 1), require unit observed at tt+l-1
        # and stable controls observed at tt-1 and tt+l-1 with D = 0
        # throughout the window [tt-1, tt+l-1].
        for k, l in enumerate(horizons):
            if l < 1 or tt + l - 1 >= T:
                continue
            t_end = tt + l - 1
            # Stable for horizon l means D = 0 on every column in [tt-1, t_end].
            window = D[:, tt - 1:t_end + 1]
            stable_l = ((~np.isnan(window)).all(axis=1)
                          & (window == 0).all(axis=1))
            sw_l = switcher & ~np.isnan(Y[:, t_end]) & ~np.isnan(Y[:, tt - 1])
            st_l = stable_l & ~np.isnan(Y[:, t_end]) & ~np.isnan(Y[:, tt - 1])
            if not sw_l.any() or not st_l.any():
                continue
            dy_sw = Y[sw_l, t_end] - Y[sw_l, tt - 1]
            dy_st = Y[st_l, t_end] - Y[st_l, tt - 1]
            contribution = dy_sw.mean() - dy_st.mean()
            w = int(sw_l.sum())
            num_Ml[k] += contribution * w
            den_Ml[k] += w

        # Placebo: at time tt-1, look back one period (tt-2). Use the
        # same units identified as switchers at tt — but compare their
        # *pre-switch* trend Y_{tt-1} - Y_{tt-2} against same-period
        # trend among units stable at D=0 across [tt-2, tt-1].
        if placebo and tt >= 2:
            d_pre2 = D[:, tt - 2]
            valid_pre = (~np.isnan(d_pre2)) & (~np.isnan(d_prev))
            stable_pre = valid_pre & (d_pre2 == 0) & (d_prev == 0)
            sw_pre = switcher & valid_pre & (d_pre2 == 0)
            sw_pre = sw_pre & ~np.isnan(Y[:, tt - 1]) & ~np.isnan(Y[:, tt - 2])
            st_pre = stable_pre & ~np.isnan(Y[:, tt - 1]) & ~np.isnan(Y[:, tt - 2])
            if sw_pre.any() and st_pre.any():
                dy_sw = Y[sw_pre, tt - 1] - Y[sw_pre, tt - 2]
                dy_st = Y[st_pre, tt - 1] - Y[st_pre, tt - 2]
                contribution = dy_sw.mean() - dy_st.mean()
                w = int(sw_pre.sum())
                num_pl += contribution * w
                den_pl += w

    att_M = num_M / den_M if den_M > 0 else float("nan")
    att_M_l = np.where(den_Ml > 0, num_Ml / np.where(den_Ml > 0, den_Ml, 1),
                          np.nan)
    att_M_placebo = num_pl / den_pl if den_pl > 0 else float("nan")
    return att_M, att_M_l, att_M_placebo, n_switchers


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cdh_did(
    y,
    treatment,
    panel_id,
    time_id,
    *,
    placebo: bool = True,
    n_boot: int = 500,
    seed: Optional[int] = None,
    horizons: Optional[tuple] = None,
) -> CdHResult:
    """de Chaisemartin-D'Haultfoeuille (2020) DID_M / DID_M^l estimator.

    Parameters
    ----------
    y : array-like, shape (N,)
        Outcome stacked long.
    treatment : array-like, shape (N,)
        Binary treatment indicator stacked long. Coerced to ``{0, 1}``.
    panel_id : array-like, shape (N,)
        Panel-unit identifier (any hashable).
    time_id : array-like, shape (N,)
        Time identifier; the algorithm uses sorted-order indices, so
        ``time_id`` need only be totally ordered.
    placebo : bool, default True
        If ``True``, run the switchers placebo and return its uniform
        p-value via the cluster bootstrap. If ``False``, ``placebo_p``
        in the returned ``CdHResult`` is ``None``.
    n_boot : int, default 500
        Number of cluster-bootstrap replications (resampling panel
        units with replacement). Used both for SEs and for the placebo
        p-value when ``placebo=True``.
    seed : int | None
        Seed for the bootstrap RNG.
    horizons : tuple of int | None
        Long-run horizons ``l`` to report. Default: ``(1, 2, 3, 4, 5)``,
        truncated to whatever the panel can support.

    Returns
    -------
    CdHResult
        Frozen dataclass with ``att_M``, ``att_M_l``, ``se_M``,
        ``se_M_l``, ``placebo_p``, ``n_switchers``, ``n_boot``,
        ``horizons``, ``names``.

    Notes
    -----
    The current implementation handles the **0 -> 1 absorbing-or-
    non-absorbing** treatment case using strict-stability comparisons
    (controls have ``D = 0`` throughout the relevant window). Leaver
    events (1 -> 0) are not yet contributed to the estimator; if your
    panel contains both joiners and leavers, the estimator will
    silently drop the latter. This is a known limitation that does not
    affect the 0 -> 1 staggered-adoption case the spec was written
    for.

    References
    ----------
    de Chaisemartin, C. and D'Haultfoeuille, X. (2020). Two-way fixed
        effects estimators with heterogeneous treatment effects. AER
        110(9), 2964-2996.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(treatment).astype(int)
    p = np.asarray(panel_id)
    t = np.asarray(time_id)
    if not (y.shape == d.shape == p.shape == t.shape) or y.ndim != 1:
        raise ValueError("y, treatment, panel_id, time_id must be 1-D arrays "
                          "of equal length")

    Y, D, units, times = _build_panel_arrays(y, d, p, t)
    T = len(times)
    if horizons is None:
        max_h = min(5, T - 1)
        horizons = tuple(range(1, max_h + 1))
    horizons = tuple(int(h) for h in horizons)

    att_M, att_M_l, att_M_placebo, n_switchers = _cdh_point(
        Y, D, horizons=horizons, placebo=placebo,
    )

    # ----- Cluster bootstrap (resample units with replacement) -----
    rng = np.random.default_rng(seed)
    L = len(horizons)
    boot_M = np.full(n_boot, np.nan)
    boot_Ml = np.full((n_boot, L), np.nan)
    boot_pl = np.full(n_boot, np.nan)
    n_units = len(units)
    for b in range(n_boot):
        idx = rng.integers(0, n_units, size=n_units)
        Yb = Y[idx]
        Db = D[idx]
        try:
            m_b, ml_b, pl_b, _ = _cdh_point(
                Yb, Db, horizons=horizons, placebo=placebo,
            )
        except Exception:
            continue
        boot_M[b] = m_b
        boot_Ml[b] = ml_b
        if placebo:
            boot_pl[b] = pl_b

    with warnings.catch_warnings():
        # All-NaN bootstrap columns (no valid replicates at horizon l)
        # are surfaced as NaN SEs; suppress numpy's "ddof <= 0" /
        # "empty slice" warnings rather than letting them leak.
        warnings.simplefilter("ignore", RuntimeWarning)
        se_M = float(np.nanstd(boot_M, ddof=0))
        se_M_l = np.nanstd(boot_Ml, axis=0, ddof=0)

    # Placebo p-value: two-sided. Compare observed |placebo statistic|
    # against the bootstrap distribution centred at zero (under H0 it's
    # mean zero). We use the fraction of |boot_pl - boot_pl.mean()|
    # >= |observed - 0| as a conservative two-sided p-value if
    # observed is the test statistic and the bootstrap measures
    # sampling variance under the null. To make this distribute as
    # uniform under the null, we use the centred bootstrap (subtract
    # the bootstrap mean) and compute a two-sided p-value via the SE.
    placebo_p: Optional[float]
    if placebo and not np.isnan(att_M_placebo):
        valid = ~np.isnan(boot_pl)
        if valid.sum() >= 2:
            se_pl = float(np.nanstd(boot_pl, ddof=1))
            if se_pl > 0:
                # z-statistic: observed / se. Under H0 (zero placebo
                # effect, parallel trends) and CLT this is ~ N(0, 1).
                from scipy.stats import norm
                z = att_M_placebo / se_pl
                placebo_p = float(2 * (1.0 - norm.cdf(abs(z))))
            else:
                placebo_p = 1.0
        else:
            placebo_p = float("nan")
    else:
        placebo_p = None

    names = ("att_M",) + tuple(f"att_M_l={h}" for h in horizons)

    return CdHResult(
        att_M=float(att_M),
        att_M_l=np.asarray(att_M_l, dtype=float),
        se_M=se_M,
        se_M_l=np.asarray(se_M_l, dtype=float),
        placebo_p=placebo_p,
        n_switchers=int(n_switchers),
        n_boot=int(n_boot),
        horizons=horizons,
        names=names,
    )


__all__ = ["cdh_did"]
