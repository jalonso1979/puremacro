"""LP-DiD: difference-in-differences event studies via local projections.

Implements the estimator of Dube, Girardi, Jordà and Taylor (DGJT),
"A Local Projections Approach to Difference-in-Differences Event
Studies" (NBER Working Paper 31184, 2023; Economica, 2025). For each
post horizon ``h ≥ 0`` LP-DiD runs the cross-sectional regression

    y_{i,t+h} − y_{i,t−1} = β_h ΔD_{it} + δ_t^h + γ_h' W_{it} + e_{it}^h

on the restricted sample of

* **newly-treated** observations — ``ΔD_{it} = 1`` (``D_{it} = 1`` and
  ``D_{i,t−1} = 0``), and
* **clean controls** — ``D_{i,t+h} = 0``: units not yet treated through
  ``t+h`` (never-treated or later-treated), which under absorbing
  treatment rules out every "forbidden" already-treated comparison.

Pre-trend coefficients for ``h = −pre_window, …, −2`` come from the
same regression with the lead outcome replaced by the lag
``y_{i,t+h}``; the clean-control condition is evaluated at
``t + max(h, 0)``, i.e. controls must be untreated through ``t``
(``h = −1`` is the base period and is included as a mechanical zero).

Weighting (DGJT §2.4). Plain OLS (``weights='vw'``) delivers a
**variance-weighted** average of the clean period-specific DiD
estimates — the convex, never-negative analogue of the TWFE weighting
(period weight ∝ n_t·p_t(1−p_t)). WLS that gives each newly-treated
observation weight 1 and each clean control in period t weight
``n_t^{treated}/n_t^{clean}`` (``weights='equal'``) delivers the
**equally-weighted ATT** across treatment events (period weight ∝
n_t^{treated}).

Simplifications vs the paper, stated up front: absorbing treatment
only (a diagnostic error is raised otherwise — the non-absorbing
variant in DGJT §2.6 additionally requires controls to be stable over
a look-back window); z-based confidence intervals; per-horizon
pre-trend t statistics rather than a joint Wald test; controls enter
linearly with no regression-adjustment refinement; and leads/lags are
positional within each unit's sorted rows (regular time spacing is
assumed, the house convention shared with ``panel_lp``).

References
----------
Dube, A., Girardi, D., Jordà, Ò. and Taylor, A.M. (2023). A local
    projections approach to difference-in-differences event studies.
    NBER Working Paper 31184 (Economica, 2025).
Callaway, B. and Sant'Anna, P.H.C. (2021). Difference-in-differences
    with multiple time periods. Journal of Econometrics 225(2), 200-230.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from .._linalg import inv_xtx

_ALLOWED_WEIGHTS = ("equal", "vw")

_EST_COLS = ["h", "beta", "se", "t", "lo", "hi",
             "n_treated", "n_clean", "n_obs", "n_periods"]


@dataclass(frozen=True)
class LPDiDResult:
    """Result of :func:`puremacro.lp.lp_did`.

    Attributes
    ----------
    estimates : pd.DataFrame
        One row per horizon (pre-trends first), columns
        ``[h, beta, se, t, lo, hi, n_treated, n_clean, n_obs,
        n_periods]``. The ``h = -1`` row is the base-period
        normalization (``beta = se = 0`` mechanically). Horizons with
        no identifying period (no newly-treated unit facing at least
        one clean control) carry ``NaN`` estimates and zero counts.
    group_weights : pd.DataFrame
        DGJT reweighting diagnostics: one row per (horizon, period)
        actually used, columns ``[h, time, n_treated, n_clean, w_vw,
        w_equal]`` where ``w_vw ∝ n_tr·n_cl/(n_tr+n_cl)`` and
        ``w_equal ∝ n_tr`` are the period shares of β_h under each
        weighting (normalized to sum to 1 within each horizon; exact
        for the no-controls specification, indicative otherwise).
    weights : str
        Weighting mode used: ``'equal'`` or ``'vw'``.
    pre_window : int
        Pre-trend window K (leads ``h = -K..-2``; 0 = none requested).
    alpha : float
        Two-sided level; CIs cover 1 − alpha.
    cluster : str
        Clustering column (``'unit'`` means the unit column).
    clean_controls : bool
        Whether the clean-control restriction was enforced.
    n_units : int
        Distinct units in the input panel.
    n_events : int
        Observed treatment switches (0→1 transitions).

    References
    ----------
    Dube, A., Girardi, D., Jordà, Ò. and Taylor, A.M. (2023). A local
        projections approach to difference-in-differences event
        studies. NBER Working Paper 31184 (Economica, 2025).
    """

    estimates: pd.DataFrame
    group_weights: pd.DataFrame
    weights: str
    pre_window: int
    alpha: float
    cluster: str
    clean_controls: bool
    n_units: int
    n_events: int

    @property
    def post(self) -> pd.DataFrame:
        """Post-treatment rows (``h >= 0``), index reset."""
        est = self.estimates
        return est[est["h"] >= 0].reset_index(drop=True)

    @property
    def pretrend(self) -> pd.DataFrame:
        """Genuine pre-trend rows (``h <= -2``; excludes the h = -1
        normalization), index reset."""
        est = self.estimates
        return est[est["h"] <= -2].reset_index(drop=True)

    def summary(self) -> str:
        """One-paragraph human-readable summary of the fit."""
        post = self.post
        pre = self.pretrend
        h0 = post[post["h"] == 0]
        h0_line = (
            f"  ATT(h=0)          : {float(h0['beta'].iloc[0]):+.4f} "
            f"(se {float(h0['se'].iloc[0]):.4f})\n" if len(h0) else ""
        )
        if len(pre) and np.isfinite(pre["t"]).any():
            max_pre_t = float(np.nanmax(np.abs(pre["t"])))
            pre_line = (f"  pre-trends        : h = -{self.pre_window}..-2, "
                        f"max |t| = {max_pre_t:.2f}\n")
        else:
            pre_line = "  pre-trends        : (none estimated)\n"
        return (
            f"LP-DiD (Dube-Girardi-Jordà-Taylor) result\n"
            f"  weights           : {self.weights}\n"
            f"  clean controls    : {self.clean_controls}\n"
            f"  units / events    : {self.n_units} / {self.n_events}\n"
            f"  post horizons     : {int(post['h'].min())}.."
            f"{int(post['h'].max())}\n"
            f"{pre_line}{h0_line}"
        )


def lp_did(
    panel: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    horizons: Iterable[int] = range(0, 13),
    *,
    pre_window: int = 4,
    clean_controls: bool = True,
    weights: str = "equal",
    controls: Optional[Sequence[str]] = None,
    cluster: str = "unit",
    alpha: float = 0.10,
) -> LPDiDResult:
    """LP-DiD event-study estimates per Dube-Girardi-Jordà-Taylor (2023).

    For each ``h`` in ``horizons`` (all ≥ 0), regresses the long
    difference ``y_{i,t+h} − y_{i,t−1}`` on the treatment-switch
    indicator ``ΔD_{it}`` with period effects, using only newly-treated
    units and clean controls (not yet treated through ``t+h``). The
    same regression at leads ``h = −pre_window..−2`` yields pre-trend
    tests "for free"; ``h = −1`` is the base period (mechanical zero).

    Parameters
    ----------
    panel : pd.DataFrame
        Long panel; one row per (unit, time).
    y, treat, unit, time : str
        Column names. ``treat`` must be a binary 0/1 **absorbing**
        indicator (once treated, always treated); a diagnostic error is
        raised for non-absorbing paths — see Notes.
    horizons : iterable of int
        Post horizons, all ≥ 0.
    pre_window : int, default 4
        K ≥ 2 estimates pre-trend leads ``h = −K..−2``; 0 skips them.
    clean_controls : bool, default True
        If False, keeps **every** non-switching observation in the
        control group — including already-treated units, i.e. the
        forbidden TWFE-style comparisons. For pedagogy and diagnostics
        only; estimates are then contaminated whenever effects are
        dynamic and heterogeneous (Goodman-Bacon 2021).
    weights : {'equal', 'vw'}, default 'equal'
        ``'equal'`` reweights so every newly-treated observation counts
        equally (the equally-weighted ATT); ``'vw'`` is plain OLS, the
        variance-weighted average with convex TWFE-style period weights
        (DGJT §2.4).
    controls : sequence of str, optional
        Additional regressor columns, entered linearly (values at the
        observation ``(i, t)``; pre-determined controls such as lagged
        outcome changes are the user's responsibility to construct).
    cluster : str, default 'unit'
        ``'unit'`` clusters standard errors by the ``unit`` column;
        any other value must name a column of ``panel`` to cluster on.
    alpha : float, default 0.10
        Two-sided level; CI half-width uses the normal critical value.

    Returns
    -------
    LPDiDResult
        Frozen dataclass with per-horizon estimates (including
        pre-trends), per-horizon treated/clean counts, and the DGJT
        period-weight diagnostics. See :class:`LPDiDResult`.

    Notes
    -----
    Non-absorbing treatment (any 1→0 switch) raises a ``ValueError``:
    supporting it requires the DGJT §2.6 variant where units qualify
    as newly treated / clean controls only relative to a stabilization
    window, which this implementation does not provide.

    References
    ----------
    Dube, A., Girardi, D., Jordà, Ò. and Taylor, A.M. (2023). A local
        projections approach to difference-in-differences event
        studies. NBER Working Paper 31184 (Economica, 2025).
    Goodman-Bacon, A. (2021). Difference-in-differences with variation
        in treatment timing. Journal of Econometrics 225(2), 254-277.
    """
    ctl = list(controls) if controls is not None else []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    if weights not in _ALLOWED_WEIGHTS:
        raise ValueError(
            f"lp_did: weights must be one of {_ALLOWED_WEIGHTS}, got {weights!r}")
    if not float(pre_window).is_integer() or pre_window < 0 or pre_window == 1:
        raise ValueError(
            "lp_did: pre_window must be 0 (skip pre-trends) or an integer >= 2 "
            f"(pre-trend leads run h = -pre_window..-2), got {pre_window!r}")
    pre_window = int(pre_window)
    if not 0 < alpha < 1:
        raise ValueError(f"lp_did: alpha must be in (0, 1), got {alpha!r}")

    hs = sorted({int(h) for h in horizons})
    if not hs:
        raise ValueError("lp_did: horizons is empty")
    if hs[0] < 0:
        raise ValueError(
            "lp_did: horizons must all be >= 0 (pre-trend leads come from "
            f"pre_window, not negative horizons); got {hs[0]}")

    needed = [unit, time, y, treat] + ctl
    if cluster != "unit":
        if cluster not in panel.columns:
            raise ValueError(
                f"lp_did: cluster must be 'unit' or a column of panel; "
                f"{cluster!r} is neither")
        needed.append(cluster)
    missing = [c for c in dict.fromkeys(needed) if c not in panel.columns]
    if missing:
        raise ValueError(f"lp_did: column(s) {missing} not in panel")

    df = panel[list(dict.fromkeys(needed))].copy()
    if df.duplicated([unit, time]).any():
        raise ValueError(
            "lp_did: panel has duplicate (unit, time) rows; aggregate them first")
    if df[treat].isna().any():
        raise ValueError(f"lp_did: treatment column {treat!r} contains NaN")
    tvals = set(pd.unique(df[treat].astype(float)))
    if not tvals <= {0.0, 1.0}:
        raise ValueError(
            f"lp_did: treatment column {treat!r} must be binary 0/1, "
            f"found values {sorted(tvals - {0.0, 1.0})[:5]}")

    df = df.sort_values([unit, time], kind="mergesort").reset_index(drop=True)
    g = df.groupby(unit, sort=False)

    D = df[treat].astype(float)
    D_prev = g[treat].shift(1).astype(float)
    reversal = (D_prev == 1.0) & (D == 0.0)
    if reversal.any():
        bad = df.loc[reversal, unit].unique()[:5].tolist()
        raise ValueError(
            f"lp_did: non-absorbing treatment (1->0 switch) detected for "
            f"unit(s) {bad}; this implementation supports absorbing treatment "
            "only. The DGJT (2023, §2.6) non-absorbing variant requires "
            "clean-control windows around each switch and is not implemented.")

    newly = ((D == 1.0) & (D_prev == 0.0)).to_numpy()
    y_base = g[y].shift(1)
    cluster_ids = (df[unit] if cluster == "unit" else df[cluster]).to_numpy()
    time_arr = df[time].to_numpy()
    z_crit = float(norm.ppf(1 - alpha / 2))

    n_units = int(df[unit].nunique())
    n_events = int(newly.sum())

    # ------------------------------------------------------------------
    # Per-horizon engine
    # ------------------------------------------------------------------
    all_h = list(range(-pre_window, -1)) if pre_window >= 2 else []
    if pre_window >= 2:
        all_h.append(-1)                    # base-period normalization row
    all_h += hs

    rows: list[dict] = []
    gw_rows: list[dict] = []

    for h in all_h:
        dy = (g[y].shift(-h) - y_base).to_numpy()
        # Clean-control condition evaluated at t + max(h, 0): untreated
        # through t+h for post horizons, untreated through t for leads.
        d_ref = (g[treat].shift(-h) if h > 0 else df[treat]).astype(float)
        if clean_controls:
            is_ctrl = (d_ref == 0.0).to_numpy()
        else:
            # Pedagogical mode: every non-switching in-sample observation,
            # including already-treated units (the forbidden comparisons).
            is_ctrl = (~newly) & d_ref.notna().to_numpy()

        mask = (newly | is_ctrl) & np.isfinite(dy)
        for c in ctl:
            mask &= df[c].notna().to_numpy()

        x_raw = newly[mask].astype(float)
        t_sub = time_arr[mask]
        tcodes, tuniq = pd.factorize(t_sub)
        n_tr_t = np.bincount(tcodes, weights=x_raw)
        n_all_t = np.bincount(tcodes)
        n_co_t = n_all_t - n_tr_t
        ident = (n_tr_t >= 1) & (n_co_t >= 1)   # periods with a real contrast

        if not ident.any():
            rows.append({"h": h, "beta": np.nan, "se": np.nan, "t": np.nan,
                         "lo": np.nan, "hi": np.nan, "n_treated": 0,
                         "n_clean": 0, "n_obs": 0, "n_periods": 0})
            continue

        keep = ident[tcodes]
        if h == -1:
            # y_{t-1} - y_{t-1} = 0 identically: report the base-period
            # normalization row (counts restricted to identifying periods)
            # without running a degenerate regression.
            n_tr = int(x_raw[keep].sum())
            n_ob = int(keep.sum())
            rows.append({"h": h, "beta": 0.0, "se": 0.0, "t": np.nan,
                         "lo": 0.0, "hi": 0.0, "n_treated": n_tr,
                         "n_clean": n_ob - n_tr, "n_obs": n_ob,
                         "n_periods": int(ident.sum())})
            continue

        x_raw = x_raw[keep]
        dy_s = dy[mask][keep]
        cl_s = cluster_ids[mask][keep]
        ctl_mat = (df.loc[mask, ctl].to_numpy(dtype=float)[keep]
                   if ctl else np.empty((keep.sum(), 0)))
        tcodes, tuniq = pd.factorize(t_sub[keep])
        n_tr_t = np.bincount(tcodes, weights=x_raw)
        n_all_t = np.bincount(tcodes)
        n_co_t = n_all_t - n_tr_t

        # Observation weights: OLS ('vw') or the DGJT reweighting that
        # equalizes each treatment event's contribution ('equal').
        if weights == "vw":
            w = np.ones(x_raw.shape[0])
        else:
            w = np.where(x_raw == 1.0, 1.0, (n_tr_t / n_co_t)[tcodes])

        # Weighted within-period demeaning == period fixed effects in WLS.
        cols = [dy_s, x_raw] + [ctl_mat[:, j] for j in range(ctl_mat.shape[1])]
        W_t = np.bincount(tcodes, weights=w)
        demeaned = []
        for v in cols:
            m_t = np.bincount(tcodes, weights=w * v) / W_t
            demeaned.append(v - m_t[tcodes])
        y_til = demeaned[0]
        X_til = np.column_stack(demeaned[1:])

        sqw = np.sqrt(w)
        Xw = X_til * sqw[:, None]
        yw = y_til * sqw
        XtX_inv = inv_xtx(Xw, name="lp_did")
        beta_vec = XtX_inv @ (Xw.T @ yw)
        resid = yw - Xw @ beta_vec

        # Cluster-robust sandwich (same convention as two_way_fe_within).
        scores = Xw * resid[:, None]
        ccodes, cuniq = pd.factorize(cl_s)
        agg = np.zeros((cuniq.size, Xw.shape[1]))
        np.add.at(agg, ccodes, scores)
        vcov = XtX_inv @ (agg.T @ agg) @ XtX_inv
        b_h = float(beta_vec[0])
        se_h = float(np.sqrt(max(vcov[0, 0], 0.0)))

        n_tr = int(x_raw.sum())
        n_ob = int(x_raw.shape[0])
        rows.append({
            "h": h, "beta": b_h, "se": se_h,
            "t": b_h / se_h if se_h > 0 else np.nan,
            "lo": b_h - z_crit * se_h, "hi": b_h + z_crit * se_h,
            "n_treated": n_tr, "n_clean": n_ob - n_tr, "n_obs": n_ob,
            "n_periods": int(ident.sum()),
        })

        # DGJT period-weight diagnostics (exact in the bivariate case).
        w_vw = n_tr_t * n_co_t / (n_tr_t + n_co_t)
        w_eq = n_tr_t.astype(float)
        w_vw = w_vw / w_vw.sum()
        w_eq = w_eq / w_eq.sum()
        for j, tv in enumerate(tuniq):
            gw_rows.append({"h": h, "time": tv,
                            "n_treated": int(n_tr_t[j]),
                            "n_clean": int(n_co_t[j]),
                            "w_vw": float(w_vw[j]),
                            "w_equal": float(w_eq[j])})

    estimates = pd.DataFrame(rows, columns=_EST_COLS)
    group_weights = pd.DataFrame(
        gw_rows, columns=["h", "time", "n_treated", "n_clean", "w_vw", "w_equal"])

    return LPDiDResult(
        estimates=estimates,
        group_weights=group_weights,
        weights=weights,
        pre_window=pre_window,
        alpha=alpha,
        cluster=cluster,
        clean_controls=clean_controls,
        n_units=n_units,
        n_events=n_events,
    )


__all__ = ["lp_did", "LPDiDResult"]
