"""Instrument-matrix construction for dynamic panel GMM.

Builds the moment-condition instrument matrix ``Z`` for difference GMM
(Arellano-Bond 1991) and the additional level-equation block for system
GMM (Blundell-Bond 1998). Supports Roodman (2009) instrument collapse
and arbitrary lag windows.

The internal data representation is a list of "panel records", each a
dict with integer panel id, sorted integer time indices, the level
outcome, and the level regressor matrix. The panel record carries
"endog/pred/exog/lag" categories on the regressor matrix so the
instrument constructor knows what lag windows apply to which column.

Notes on shapes
---------------
- ``y`` and the regressor matrix in a panel record are at the LEVEL,
  indexed by sorted unique time stamps.
- Difference rows live at time stamps ``t`` for which both ``t`` and
  ``t-1`` are present in the panel ("usable difference time stamps").
  The first usable difference is at the second observed time stamp at
  the earliest, and only when ``t-1`` (in the integer-time sense) is
  also observed.
- For lag-window construction we treat lag ``l`` of regressor ``w``
  for the difference at time ``t`` as the LEVEL value of ``w`` at
  ``t - l`` (Arellano-Bond 1991 page 280). It must be observed in the
  panel.

References
----------
Arellano, M. and Bond, S. (1991). Some tests of specification for panel
    data. Review of Economic Studies 58(2), 277-297.
Blundell, R. and Bond, S. (1998). Initial conditions and moment
    restrictions in dynamic panel data models. Journal of Econometrics
    87(1), 115-143.
Roodman, D. (2009). How to do xtabond2: an introduction to difference
    and system GMM in Stata. Stata Journal 9(1), 86-136.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------
# Long-format → panel-record conversion
# ---------------------------------------------------------------------


def _build_panel_records(
    y: np.ndarray,
    panel_id: np.ndarray,
    time_id: np.ndarray,
    X_endog: np.ndarray | None,
    X_pred: np.ndarray | None,
    X_exog: np.ndarray | None,
) -> list:
    """Group long-format data into per-panel records.

    Returns a list of dicts:
      ``{"id", "t" (sorted ints), "y", "Xe", "Xp", "Xx"}``
    where the X matrices may be empty (n × 0).
    """
    y = np.asarray(y, dtype=float).ravel()
    panel_id = np.asarray(panel_id).ravel()
    time_id = np.asarray(time_id, dtype=np.int64).ravel()
    if not (len(y) == len(panel_id) == len(time_id)):
        raise ValueError(
            "y, panel_id, time_id must have the same length "
            f"(got {len(y)}, {len(panel_id)}, {len(time_id)})"
        )
    n = len(y)

    def _check(X, name):
        if X is None:
            return np.zeros((n, 0))
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        if X.shape[0] != n:
            raise ValueError(
                f"{name} has {X.shape[0]} rows; expected {n} (matching y)."
            )
        return X

    Xe = _check(X_endog, "X_endog")
    Xp = _check(X_pred, "X_pred")
    Xx = _check(X_exog, "X_exog")

    # group by panel
    uniq_panels = np.unique(panel_id)
    records = []
    for pid in uniq_panels:
        mask = panel_id == pid
        t = time_id[mask]
        order = np.argsort(t)
        t_sorted = t[order]
        # check no duplicate (panel, time)
        if len(np.unique(t_sorted)) != len(t_sorted):
            raise ValueError(
                f"Panel {pid!r} has duplicate time_id values."
            )
        rec = {
            "id": pid,
            "t": t_sorted,
            "y": y[mask][order],
            "Xe": Xe[mask][order],
            "Xp": Xp[mask][order],
            "Xx": Xx[mask][order],
        }
        records.append(rec)
    return records


# ---------------------------------------------------------------------
# Per-panel difference rows: which t's give a usable Δ-row
# ---------------------------------------------------------------------


def _usable_diff_times(t_sorted: np.ndarray) -> np.ndarray:
    """Return the times ``t`` for which ``t-1`` is also in ``t_sorted``.

    ``t_sorted`` is expected to be a sorted array of distinct integer
    time indices for one panel.
    """
    if len(t_sorted) < 2:
        return np.empty(0, dtype=t_sorted.dtype)
    tset = set(t_sorted.tolist())
    keep = [int(t) for t in t_sorted if (int(t) - 1) in tset]
    return np.array(keep, dtype=t_sorted.dtype)


def _level_at(rec: dict, t: int, key: str, col: int) -> float | None:
    """Return the level of column ``col`` of regressor matrix ``key``
    at time ``t`` for panel ``rec``, or ``None`` if not observed.
    """
    idx = np.where(rec["t"] == t)[0]
    if len(idx) == 0:
        return None
    return float(rec[key][idx[0], col])


def _y_level_at(rec: dict, t: int) -> float | None:
    idx = np.where(rec["t"] == t)[0]
    if len(idx) == 0:
        return None
    return float(rec["y"][idx[0]])


# ---------------------------------------------------------------------
# Lag-window helpers
# ---------------------------------------------------------------------


def _lag_window(
    lo: int, hi: int | None, t: int, t_min: int, t_max_include: int | None = None
) -> list:
    """Return the list of lags ``l`` in ``[lo, hi]`` such that ``t - l``
    is in ``[t_min, t_max_include]`` (defaults to ``+inf``).
    """
    if hi is None:
        hi = t - t_min
    out = []
    for l in range(lo, hi + 1):
        s = t - l
        if s < t_min:
            continue
        if t_max_include is not None and s > t_max_include:
            continue
        out.append(l)
    return out


# ---------------------------------------------------------------------
# Difference-equation instrument matrix
# ---------------------------------------------------------------------


def build_diff_instruments(
    records: list,
    *,
    lag_dep_var: int,
    n_endog: int,
    n_pred: int,
    n_exog: int,
    gmm_lag_window: tuple,
    collapse: bool,
) -> tuple:
    """Build the difference-equation instrument matrix ``Z`` for AB GMM.

    Parameters
    ----------
    records : list of panel records.
    lag_dep_var : number of lags of ``y`` on the RHS (e.g. 1 or 2).
        The maximally-lagged y on the RHS is treated as endogenous; all
        lagged-y RHS terms enter the lag-window block via the level of
        ``y`` at ``t - 2, t - 3, ...`` (Arellano-Bond 1991).
    n_endog, n_pred, n_exog : counts of regressors of each kind.
    gmm_lag_window : ``(lo, hi)`` for endog and one-shifted ``(lo-1, hi-1)``
        for predetermined; ``hi=None`` means no upper bound.
    collapse : Roodman (2009) instrument collapse — one column per lag
        rather than one per (lag, t).

    Returns
    -------
    Z : ndarray, shape (n_diff_rows, n_instruments)
    diff_rows : list of dicts
        Per-row metadata:
        ``{"panel": pid, "t": int, "row_in_Z": int}``.
        Used downstream for residual reconstruction and AR tests.
    """
    lo, hi = gmm_lag_window
    if lo < 1:
        raise ValueError(f"gmm_lag_window low end must be >= 1 (got {lo}).")

    # First pass: collect usable diff rows per panel.
    panel_blocks = []
    n_diff_total = 0
    for rec in records:
        usable = _usable_diff_times(rec["t"])
        if len(usable) == 0:
            continue
        panel_blocks.append((rec, usable))
        n_diff_total += len(usable)

    if n_diff_total == 0:
        raise ValueError(
            "No usable difference rows across the panel "
            "(every panel has fewer than 2 consecutive time periods)."
        )

    # Determine the maximum lag we'll ever try for each "GMM-style"
    # category (endog or pred). With collapse=True: one column per lag.
    # With collapse=False: one column per (lag, calendar t) pair where
    # any panel actually populated that combo (Holtz-Eakin et al. 1988
    # uncollapsed style).
    #
    # We'll build instrument columns category-by-category.

    # Compute the global time range
    all_t = np.concatenate([rec["t"] for rec in records]) if records else np.empty(0, dtype=np.int64)
    if len(all_t) == 0:
        raise ValueError("No data.")
    global_t_min = int(all_t.min())

    # gather usable diff times across all panels (for uncollapsed columns)
    usable_diff_times_global = sorted({
        int(t)
        for (_rec, usable) in panel_blocks
        for t in usable
    })

    # Helper to build the block of GMM-style instruments for one
    # regressor (matrix key + column).
    def _gmm_block(reg_key: str, col: int, low: int, high) -> tuple:
        """Return (Z_block (n_diff_total × ncols), col_labels: list[str])."""
        # Determine the maximum lag actually realisable across the data.
        # For each diff row at time t, the maximum lag is t - min(t' available
        # in this panel <= t - low). Practically: per-panel max lag is
        # ``t - rec_t_min``.
        if collapse:
            # one column per lag in [low, high_eff]
            # high_eff = max over rows of (t - rec_t_min)
            high_eff = 0
            for rec, usable in panel_blocks:
                rec_t_min = int(rec["t"].min())
                for t in usable:
                    he = int(t) - rec_t_min
                    if he > high_eff:
                        high_eff = he
            if high is not None:
                high_eff = min(high_eff, high)
            if high_eff < low:
                return np.zeros((n_diff_total, 0)), []
            ncols = high_eff - low + 1
            Z_block = np.zeros((n_diff_total, ncols))
            offset = 0
            for rec, usable in panel_blocks:
                for t in usable:
                    for j, l in enumerate(range(low, high_eff + 1)):
                        s = int(t) - l
                        v = _level_at(rec, s, reg_key, col)
                        # missing → 0 (Holtz-Eakin et al. 1988 convention)
                        Z_block[offset, j] = 0.0 if v is None else v
                    offset += 1
            labels = [f"{reg_key}{col}_L{l}" for l in range(low, high_eff + 1)]
            return Z_block, labels
        else:
            # uncollapsed: one column per (lag l, calendar time t)
            # we iterate panels in fixed order; rows are in panel-blocks order
            # collect the (l, t) basis from globally observed usable diff times
            basis = []  # (lag, t)
            for t in usable_diff_times_global:
                lo_eff = low
                hi_eff = high if high is not None else (int(t) - global_t_min)
                hi_eff = min(hi_eff, int(t) - global_t_min)
                if hi_eff < lo_eff:
                    continue
                for l in range(lo_eff, hi_eff + 1):
                    basis.append((l, int(t)))
            if not basis:
                return np.zeros((n_diff_total, 0)), []
            ncols = len(basis)
            Z_block = np.zeros((n_diff_total, ncols))
            t_to_col = {(l, t): j for j, (l, t) in enumerate(basis)}
            offset = 0
            for rec, usable in panel_blocks:
                rec_t_min = int(rec["t"].min())
                for t in usable:
                    t_int = int(t)
                    for (l, t_basis), j in t_to_col.items():
                        if t_basis != t_int:
                            continue
                        s = t_int - l
                        if s < rec_t_min:
                            continue
                        v = _level_at(rec, s, reg_key, col)
                        if v is not None:
                            Z_block[offset, j] = v
                    offset += 1
            labels = [f"{reg_key}{col}_L{l}@t{t}" for (l, t) in basis]
            return Z_block, labels

    blocks = []
    labels = []

    # 1) lagged-y as endogenous (lag_dep_var on RHS)
    # the GMM-style instrument block uses lags (lo, hi) of LEVEL y.
    # Use the helper with reg_key="y_level".
    # Inject a synthetic "y_level" matrix into records lazily by
    # constructing it from rec["y"].
    for rec in records:
        # store y as shape (n,1) under a synthetic key "_yL"
        rec["_yL"] = rec["y"][:, None]

    block_y, lab_y = _gmm_block("_yL", 0, lo, hi)
    if block_y.shape[1] > 0:
        blocks.append(block_y)
        labels.extend(lab_y)

    # 2) X_endog (lag window same as lagged-y)
    for j in range(n_endog):
        b, l = _gmm_block("Xe", j, lo, hi)
        if b.shape[1] > 0:
            blocks.append(b)
            labels.extend(l)

    # 3) X_pred (lag window shifted by -1: starts at lo-1, but at least 1)
    pred_lo = max(1, lo - 1)
    pred_hi = (hi - 1) if (hi is not None) else None
    for j in range(n_pred):
        b, l = _gmm_block("Xp", j, pred_lo, pred_hi)
        if b.shape[1] > 0:
            blocks.append(b)
            labels.extend(l)

    # 4) X_exog: strictly exogenous → use ΔX_exog itself as instrument
    # (one column per exog regressor). This goes in the "IV-style"
    # block, NOT the GMM-style block.
    if n_exog > 0:
        Z_exog = np.zeros((n_diff_total, n_exog))
        offset = 0
        for rec, usable in panel_blocks:
            for t in usable:
                t_int = int(t)
                for j in range(n_exog):
                    v_t = _level_at(rec, t_int, "Xx", j)
                    v_tm1 = _level_at(rec, t_int - 1, "Xx", j)
                    if v_t is not None and v_tm1 is not None:
                        Z_exog[offset, j] = v_t - v_tm1
                offset += 1
        blocks.append(Z_exog)
        labels.extend([f"Xx{j}_diff" for j in range(n_exog)])

    if not blocks:
        raise ValueError(
            "No instruments could be constructed; check lag windows and "
            "regressor categorisation."
        )

    # cleanup
    for rec in records:
        rec.pop("_yL", None)

    Z = np.concatenate(blocks, axis=1)

    # diff_rows metadata
    diff_rows = []
    offset = 0
    for rec, usable in panel_blocks:
        for t in usable:
            diff_rows.append({"panel": rec["id"], "t": int(t), "row": offset})
            offset += 1

    return Z, diff_rows, labels


# ---------------------------------------------------------------------
# Difference-equation regressor matrix and outcome
# ---------------------------------------------------------------------


def build_diff_design(
    records: list,
    diff_rows: list,
    *,
    lag_dep_var: int,
    n_endog: int,
    n_pred: int,
    n_exog: int,
) -> tuple:
    """Build ``Δy`` (n_rows,) and ``ΔX`` (n_rows, k) for the difference
    equation, as well as a list of regressor names (length k).

    Layout of ``ΔX`` columns (in this fixed order):
      1. ``Δy_{t-1}, Δy_{t-2}, ..., Δy_{t-lag_dep_var}``
      2. ``ΔX_endog`` columns
      3. ``ΔX_pred`` columns
      4. ``ΔX_exog`` columns
    """
    n_rows = len(diff_rows)
    k = lag_dep_var + n_endog + n_pred + n_exog
    dy = np.full(n_rows, np.nan)
    dX = np.full((n_rows, k), np.nan)

    # build a quick lookup: panel id -> rec
    by_id = {rec["id"]: rec for rec in records}

    for r, drow in enumerate(diff_rows):
        rec = by_id[drow["panel"]]
        t = drow["t"]
        # Δy_t = y_t - y_{t-1}
        y_t = _y_level_at(rec, t)
        y_tm1 = _y_level_at(rec, t - 1)
        if y_t is None or y_tm1 is None:
            continue  # should never happen given diff_rows construction
        dy[r] = y_t - y_tm1

        col = 0
        # lagged Δy on RHS: Δy_{t-1} = y_{t-1} - y_{t-2}, etc.
        for p in range(1, lag_dep_var + 1):
            y_a = _y_level_at(rec, t - p)
            y_b = _y_level_at(rec, t - p - 1)
            if y_a is None or y_b is None:
                dX[r, col] = np.nan
            else:
                dX[r, col] = y_a - y_b
            col += 1

        # ΔX_endog
        for j in range(n_endog):
            v_t = _level_at(rec, t, "Xe", j)
            v_tm1 = _level_at(rec, t - 1, "Xe", j)
            if v_t is None or v_tm1 is None:
                dX[r, col] = np.nan
            else:
                dX[r, col] = v_t - v_tm1
            col += 1

        # ΔX_pred
        for j in range(n_pred):
            v_t = _level_at(rec, t, "Xp", j)
            v_tm1 = _level_at(rec, t - 1, "Xp", j)
            if v_t is None or v_tm1 is None:
                dX[r, col] = np.nan
            else:
                dX[r, col] = v_t - v_tm1
            col += 1

        # ΔX_exog
        for j in range(n_exog):
            v_t = _level_at(rec, t, "Xx", j)
            v_tm1 = _level_at(rec, t - 1, "Xx", j)
            if v_t is None or v_tm1 is None:
                dX[r, col] = np.nan
            else:
                dX[r, col] = v_t - v_tm1
            col += 1

    # names
    names = []
    for p in range(1, lag_dep_var + 1):
        names.append(f"L{p}.y")
    names.extend([f"endog_{j}" for j in range(n_endog)])
    names.extend([f"pred_{j}" for j in range(n_pred)])
    names.extend([f"exog_{j}" for j in range(n_exog)])

    return dy, dX, names


# ---------------------------------------------------------------------
# H matrix used for the one-step weighting (Arellano-Bond 1991, eq. (8))
# ---------------------------------------------------------------------


def build_H_block(diff_rows: list) -> np.ndarray:
    """Construct the block-diagonal first-difference covariance ``H``.

    For each panel block (list of consecutive usable diff rows), ``H_i``
    is a banded matrix with ``2`` on the diagonal and ``-1`` on the
    immediate off-diagonals — corresponding to ``Cov(Δε)`` under iid
    homoskedastic ``ε``.

    Returns ``H`` as an (n × n) numpy array.
    """
    n = len(diff_rows)
    H = np.zeros((n, n))
    # group rows by panel using contiguous blocks
    if n == 0:
        return H
    start = 0
    cur_panel = diff_rows[0]["panel"]
    cur_times = [diff_rows[0]["t"]]
    starts_ends = []
    for r in range(1, n):
        if diff_rows[r]["panel"] == cur_panel:
            cur_times.append(diff_rows[r]["t"])
        else:
            starts_ends.append((start, r, cur_times))
            start = r
            cur_panel = diff_rows[r]["panel"]
            cur_times = [diff_rows[r]["t"]]
    starts_ends.append((start, n, cur_times))

    for s, e, times in starts_ends:
        m = e - s
        H[s:e, s:e] = 2.0 * np.eye(m)
        for k in range(m - 1):
            # only set off-diag if times are consecutive (the diff rows
            # at t and t+1 share ε_t; otherwise no covariance)
            if times[k + 1] - times[k] == 1:
                H[s + k, s + k + 1] = -1.0
                H[s + k + 1, s + k] = -1.0
    return H


# ---------------------------------------------------------------------
# Public wrapper used by ab_gmm / bb_gmm
# ---------------------------------------------------------------------


def build_instruments(
    y: np.ndarray,
    panel_id: np.ndarray,
    time_id: np.ndarray,
    *,
    lag_dep_var: int = 1,
    X_endog: np.ndarray | None = None,
    X_pred: np.ndarray | None = None,
    X_exog: np.ndarray | None = None,
    gmm_lag_window: tuple = (2, None),
    collapse: bool = True,
) -> dict:
    """Build everything needed to estimate the difference equation by
    GMM: instrument matrix ``Z``, design matrix ``ΔX``, outcome ``Δy``,
    weighting helper ``H``, and per-row metadata.

    Returns
    -------
    dict with keys:
        ``Z``, ``X_diff``, ``y_diff``, ``H``, ``diff_rows``,
        ``records``, ``names``, ``instr_labels``.
    """
    if lag_dep_var < 1:
        raise ValueError("lag_dep_var must be >= 1.")

    records = _build_panel_records(y, panel_id, time_id, X_endog, X_pred, X_exog)
    n_endog = records[0]["Xe"].shape[1] if records else 0
    n_pred = records[0]["Xp"].shape[1] if records else 0
    n_exog = records[0]["Xx"].shape[1] if records else 0

    Z, diff_rows, instr_labels = build_diff_instruments(
        records,
        lag_dep_var=lag_dep_var,
        n_endog=n_endog,
        n_pred=n_pred,
        n_exog=n_exog,
        gmm_lag_window=gmm_lag_window,
        collapse=collapse,
    )

    dy, dX, names = build_diff_design(
        records,
        diff_rows,
        lag_dep_var=lag_dep_var,
        n_endog=n_endog,
        n_pred=n_pred,
        n_exog=n_exog,
    )

    # drop rows where dy or any dX entry is NaN (gap-induced unusable)
    keep = np.isfinite(dy) & np.all(np.isfinite(dX), axis=1)
    Z = Z[keep]
    dy = dy[keep]
    dX = dX[keep]
    diff_rows = [d for k, d in zip(keep, diff_rows) if k]

    if len(diff_rows) == 0:
        raise ValueError(
            "After dropping rows with missing differences, no usable "
            "observations remain. Check lag_dep_var and the time gaps."
        )

    H = build_H_block(diff_rows)

    return {
        "Z": Z,
        "X_diff": dX,
        "y_diff": dy,
        "H": H,
        "diff_rows": diff_rows,
        "records": records,
        "names": names,
        "instr_labels": instr_labels,
        "n_endog": n_endog,
        "n_pred": n_pred,
        "n_exog": n_exog,
        "lag_dep_var": lag_dep_var,
    }


__all__ = [
    "build_instruments",
    "build_diff_instruments",
    "build_diff_design",
    "build_H_block",
]
