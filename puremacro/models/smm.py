"""SMM moment loader and objective for the DMP regime-dependent estimation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MomentVector:
    time_series: np.ndarray              # shape (36,) — 9h × 2 regimes × 2 outcomes (u, v)
    cross_section: np.ndarray            # shape (6,)  — 3h × 2 regimes × (Q4-Q1)
    weights_time_series: np.ndarray      # diagonal weights for TS block (inverse-variance)
    weights_cross_section: np.ndarray    # diagonal weights for XS block
    horizons_ts: tuple[int, ...] = tuple(range(0, 9))
    horizons_xs: tuple[int, ...] = (6, 12, 18)


def load_empirical_moments(root: Path) -> MomentVector:
    """Extract empirical moments from existing LP output tables.

    Time-series block: per-regime log(urate) and log(v) IRFs at h=0..8.
    Cross-section block: AIOE-quartile (Q4 - Q1) differences in state-Bartik
    log_urate response at h ∈ {6, 12, 18} months under each regime.

    If a required CSV is absent, the corresponding moments are set to NaN
    and their weights to zero, so the SMM objective ignores them. This lets
    the loader run before all upstream LPs have been built.
    """
    h_ts = list(range(0, 9))
    n_ts = 4 * len(h_ts)  # u_H, u_L, v_H, v_L
    ts_vec = np.full(n_ts, np.nan)
    ts_w = np.zeros(n_ts)

    # log-urate IRFs
    # NB: the misleadingly-named 36_lp_state_dep_rate.csv is in **level** urate
    # (percentage points). The correct log-urate file is logurate_main_lp.csv,
    # filtered to state == 'rate_state' (the FFR-deviation regime, matching the
    # model's regime indicator in _free_entry_residual).
    ts_u_csv = root / "notebooks" / "output_tables" / "logurate_main_lp.csv"
    if ts_u_csv.exists():
        df = pd.read_csv(ts_u_csv)
        df = df[df["state"] == "rate_state"]  # filter to FFR-deviation regime
        df_h = df[df["h"].isin(h_ts)].set_index("h").sort_index()
        for i, h in enumerate(h_ts):
            if h in df_h.index:
                row = df_h.loc[h]
                ts_vec[i] = float(row["beta_H"])
                ts_vec[i + len(h_ts)] = float(row["beta_L"])
                if "se_H" in df_h.columns and "se_L" in df_h.columns:
                    seH = float(row["se_H"])
                    seL = float(row["se_L"])
                    if seH > 0:
                        ts_w[i] = 1.0 / seH ** 2
                    if seL > 0:
                        ts_w[i + len(h_ts)] = 1.0 / seL ** 2

    # log-vacancies IRFs (may not exist yet, Task 2.3 builds it)
    ts_v_csv = root / "notebooks" / "output_tables" / "36_lp_state_dep_log_vacancies.csv"
    if ts_v_csv.exists():
        df = pd.read_csv(ts_v_csv)
        df_h = df[df["h"].isin(h_ts)].set_index("h").sort_index()
        for i, h in enumerate(h_ts):
            if h in df_h.index:
                row = df_h.loc[h]
                ts_vec[i + 2 * len(h_ts)] = float(row["beta_H"])
                ts_vec[i + 3 * len(h_ts)] = float(row["beta_L"])
                if "se_H" in df_h.columns and "se_L" in df_h.columns:
                    seH = float(row["se_H"])
                    seL = float(row["se_L"])
                    if seH > 0:
                        ts_w[i + 2 * len(h_ts)] = 1.0 / seH ** 2
                    if seL > 0:
                        ts_w[i + 3 * len(h_ts)] = 1.0 / seL ** 2

    # Cross-section block — urate AIOE-quartile diffs (Task 2.3 builds the CSV)
    h_xs = [6, 12, 18]
    n_xs = 2 * len(h_xs)
    xs_vec = np.full(n_xs, np.nan)
    xs_w = np.zeros(n_xs)

    xs_csv = root / "notebooks" / "output_tables" / "41_state_bartik_urate_quartile.csv"
    if xs_csv.exists():
        df = pd.read_csv(xs_csv)
        idx = 0
        for regime in ["tight", "loose"]:
            for h in h_xs:
                sub = df[(df["h_months"] == h) & (df["regime"] == regime)]
                try:
                    q4 = sub.loc[sub["aioe_quartile"] == 4]
                    q1 = sub.loc[sub["aioe_quartile"] == 1]
                    b4 = float(q4["beta"].iloc[0])
                    b1 = float(q1["beta"].iloc[0])
                    se4 = float(q4["se"].iloc[0])
                    se1 = float(q1["se"].iloc[0])
                    xs_vec[idx] = b4 - b1
                    var_diff = se4 ** 2 + se1 ** 2
                    if var_diff > 0:
                        xs_w[idx] = 1.0 / var_diff
                except (IndexError, KeyError):
                    pass  # leave as NaN/0
                idx += 1

    return MomentVector(
        time_series=ts_vec,
        cross_section=xs_vec,
        weights_time_series=ts_w,
        weights_cross_section=xs_w,
    )


def load_empirical_moments_monthly(
    root: Path,
    ts_horizons_months: tuple[int, ...] = (0, 3, 6, 9, 12, 15, 18, 21, 24),
) -> MomentVector:
    """Build a MomentVector from the monthly empirical LP at quarterly-spaced horizons.

    Source: ``notebooks/output_tables/paper_monthly_lp.csv`` — monthly LP of
    log-urate on the LTUI shock under the DGS2-deviation regime split (H = tight
    / high rate, L = loose / low rate), horizons h=0..24 months.  Sampled at
    ``ts_horizons_months`` (default: every 3rd month, i.e. 0, 3, 6, ..., 24)
    so the resulting 9-horizon vector aligns with the model's quarterly h=0..8
    grid used by ``smm_objective`` / ``theoretical_moments``.

    The monthly LP does not include a vacancy series, so the v_H and v_L blocks
    of the time-series vector are set to NaN with zero weight — the SMM objective
    ignores them automatically.

    Under this moment set, phi_v > 0 IS identified (commit 27890a6 diagnostic).
    The monthly H-regime IRF has a pronounced hump (peak ratio ≈ 6.7 at h≈5
    months) that the phi_v vacancy-cost channel fits better than the standard
    DMP (phi_v=0).  Contrast with ``load_empirical_moments()``, which uses the
    quarterly LP and strictly disfavours phi_v > 0.

    The cross-section block is shared with ``load_empirical_moments()`` (same
    ``41_state_bartik_urate_quartile.csv`` source).  If that CSV is absent, XS
    moments are zeroed out (as in the baseline loader).
    """
    n_horizons = len(ts_horizons_months)
    h_q_list = list(range(n_horizons))  # 0..8 (quarterly model horizons)

    # 4 blocks: u_H, u_L, v_H, v_L — each of length n_horizons
    n_ts = 4 * n_horizons
    ts_vec = np.full(n_ts, np.nan)
    ts_w = np.zeros(n_ts)

    # Offsets for each block in the flat vector (mirrors load_empirical_moments order)
    OFF_uH = 0
    OFF_uL = n_horizons
    OFF_vH = 2 * n_horizons
    OFF_vL = 3 * n_horizons

    # -- u_H and u_L from paper_monthly_lp.csv --
    monthly_csv = root / "notebooks" / "output_tables" / "paper_monthly_lp.csv"
    if monthly_csv.exists():
        df_m = pd.read_csv(monthly_csv)
        # Keep only the requested monthly horizons and sort
        df_sel = df_m[df_m["h"].isin(ts_horizons_months)].copy()
        df_sel = df_sel.set_index("h").sort_index()

        for qi, h_m in enumerate(sorted(ts_horizons_months)):
            if h_m not in df_sel.index:
                continue
            row = df_sel.loc[h_m]
            ts_vec[OFF_uH + qi] = float(row["beta_H"])
            ts_vec[OFF_uL + qi] = float(row["beta_L"])
            seH = float(row.get("se_H", 0.0) or 0.0)
            seL = float(row.get("se_L", 0.0) or 0.0)
            if seH > 0:
                ts_w[OFF_uH + qi] = 1.0 / seH ** 2
            if seL > 0:
                ts_w[OFF_uL + qi] = 1.0 / seL ** 2
    # v_H and v_L remain NaN with zero weight (no vacancy LP in paper_monthly_lp.csv)

    # -- Cross-section block (same as load_empirical_moments) --
    h_xs = [6, 12, 18]
    n_xs = 2 * len(h_xs)
    xs_vec = np.full(n_xs, np.nan)
    xs_w = np.zeros(n_xs)

    xs_csv = root / "notebooks" / "output_tables" / "41_state_bartik_urate_quartile.csv"
    if xs_csv.exists():
        df = pd.read_csv(xs_csv)
        idx = 0
        for regime in ["tight", "loose"]:
            for h in h_xs:
                sub = df[(df["h_months"] == h) & (df["regime"] == regime)]
                try:
                    q4 = sub.loc[sub["aioe_quartile"] == 4]
                    q1 = sub.loc[sub["aioe_quartile"] == 1]
                    b4 = float(q4["beta"].iloc[0])
                    b1 = float(q1["beta"].iloc[0])
                    se4 = float(q4["se"].iloc[0])
                    se1 = float(q1["se"].iloc[0])
                    xs_vec[idx] = b4 - b1
                    var_diff = se4 ** 2 + se1 ** 2
                    if var_diff > 0:
                        xs_w[idx] = 1.0 / var_diff
                except (IndexError, KeyError):
                    pass
                idx += 1

    return MomentVector(
        time_series=ts_vec,
        cross_section=xs_vec,
        weights_time_series=ts_w,
        weights_cross_section=xs_w,
    )


from typing import Literal, cast

from puremacro.models.dmp_regime_dependent import DMPParameters, dmp_irf


def theoretical_moments(
    p: DMPParameters, *,
    sigma_shock: float = 1.0,
    aioe_quartile_scaling: tuple[float, float] = (-1.5, 1.5),
) -> MomentVector:
    """Compute model-implied moments matching the empirical block shape.

    Time-series block: per-regime IRFs of log_u and log_v at h=0..8.
    Cross-section block: model-implied (Q4 - Q1) AIOE-quartile differences
    in log_u at h = 6, 12, 18 months under each regime. AIOE enters by
    scaling the LTUI shock multiplicatively at the firm-aggregate level.
    Quartile scaling defaults to (-1.5, +1.5) standard deviations — Q1 vs Q4
    AIOE z-score difference roughly matches the empirical state-AIOE
    distribution.
    """
    irf_H = dmp_irf(p, sigma_shock=sigma_shock, regime="H", horizon=8)
    irf_L = dmp_irf(p, sigma_shock=sigma_shock, regime="L", horizon=8)
    ts_vec = np.concatenate([
        irf_H.log_u, irf_L.log_u, irf_H.log_v, irf_L.log_v,
    ])

    # Cross-section: Q4 - Q1 log_u difference under each regime at three
    # horizons. Map h_months -> h_quarters: 6m≈2q, 12m≈4q, 18m≈6q. We use
    # log_u (model variable) to match the empirical cross-section's log-urate.
    h_xs_quarters = [2, 4, 6]
    xs_pieces = []
    for regime in cast(list[Literal["H", "L"]], ["H", "L"]):
        irf_q1 = dmp_irf(p, sigma_shock=sigma_shock * aioe_quartile_scaling[0],
                          regime=regime, horizon=max(h_xs_quarters))
        irf_q4 = dmp_irf(p, sigma_shock=sigma_shock * aioe_quartile_scaling[1],
                          regime=regime, horizon=max(h_xs_quarters))
        for h in h_xs_quarters:
            xs_pieces.append(irf_q4.log_u[h] - irf_q1.log_u[h])
    xs_vec = np.array(xs_pieces)

    return MomentVector(
        time_series=ts_vec,
        cross_section=xs_vec,
        weights_time_series=np.ones_like(ts_vec),
        weights_cross_section=np.ones_like(xs_vec),
    )


def smm_objective(p: DMPParameters, empirical: MomentVector) -> float:
    """Weighted squared-error SMM objective.

    Returns sum_i w_i (theoretical_i - empirical_i)^2 over both blocks.
    NaN moments in the empirical vector are ignored (their weight is zero).
    """
    try:
        th = theoretical_moments(p)
    except Exception:
        return 1e10  # parameter infeasible (e.g., DMP admissibility violation)

    # Time-series term — skip any empirical NaN moments (zero weight)
    ts_err = th.time_series - empirical.time_series
    ts_finite = np.isfinite(ts_err)
    ts_term = float(
        np.sum(empirical.weights_time_series[ts_finite] * (ts_err[ts_finite] ** 2))
    )

    # Cross-section term
    xs_err = th.cross_section - empirical.cross_section
    xs_finite = np.isfinite(xs_err)
    xs_term = float(
        np.sum(empirical.weights_cross_section[xs_finite] * (xs_err[xs_finite] ** 2))
    )

    return ts_term + xs_term
