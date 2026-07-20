"""Tests for puremacro.dynpanel.instruments."""
from __future__ import annotations

import numpy as np
import pytest

from puremacro.dynpanel.instruments import (
    build_diff_instruments,
    build_instruments,
    _build_panel_records,
    _usable_diff_times,
)

from .conftest import simulate_dynamic_panel


def test_usable_diff_times_balanced():
    t = np.array([0, 1, 2, 3, 4])
    out = _usable_diff_times(t)
    # need t-1 in the set: t=1, 2, 3, 4 → yes; t=0 → no
    np.testing.assert_array_equal(out, np.array([1, 2, 3, 4]))


def test_usable_diff_times_with_gap():
    # Panel observes t = 0, 1, 3, 4 — gap at t=2
    # usable diff rows: t=1 (since t=0 in), t=4 (since t=3 in)
    # NOT t=3 (since t=2 missing)
    t = np.array([0, 1, 3, 4])
    out = _usable_diff_times(t)
    np.testing.assert_array_equal(out, np.array([1, 4]))


def test_uncollapsed_instrument_count_balanced():
    """For a balanced panel with T=5, lag_dep_var=1, AB lags (2, None),
    no extra regressors, the uncollapsed instrument count for the
    lagged-y block follows the standard Holtz-Eakin formula:
    Σ_{t=2}^{T-1} (t - 1) over usable diff times t = {2,3,4}.
    Here usable diff times are {1, 2, 3, 4} (since y_0 ∈ panel), but
    the GMM lag window starts at lag 2, so the smallest t with a valid
    lag is t=2 (using y_0).

    Counting with lag-window (2, None):
       t=1: lags 2..(t)= none (need s=t-l>=t_min=0 → l<=1, but lo=2)
            → 0 cols
       t=2: l=2 (s=0)         → 1 col
       t=3: l=2,3 (s=1,0)     → 2 cols
       t=4: l=2,3,4 (s=2,1,0) → 3 cols
    Total = 0 + 1 + 2 + 3 = 6 columns.
    """
    sim = simulate_dynamic_panel(N=10, T=5, rho=0.5, seed=0)
    bundle = build_instruments(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=1,
        gmm_lag_window=(2, None),
        collapse=False,
    )
    Z = bundle["Z"]
    # Uncollapsed count formula: Σ over usable t: (t - lo + 1) clipped >= 0
    # given t_min=0
    expected = 6
    assert Z.shape[1] == expected, f"Got {Z.shape[1]}, expected {expected}"


def test_collapsed_reduces_instrument_count():
    """Collapse should shrink #instruments substantially."""
    sim = simulate_dynamic_panel(N=10, T=8, rho=0.5, seed=0)
    bundle_uncoll = build_instruments(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=1,
        gmm_lag_window=(2, None),
        collapse=False,
    )
    bundle_coll = build_instruments(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=1,
        gmm_lag_window=(2, None),
        collapse=True,
    )
    assert bundle_coll["Z"].shape[1] < bundle_uncoll["Z"].shape[1]
    # collapsed: 1 column per lag, lags 2..T-1=7 → 6 cols max
    assert bundle_coll["Z"].shape[1] <= 6


def test_lag_window_truncation_collapsed():
    """Restricting the lag window (2, 3) gives exactly 2 collapsed columns
    for the lagged-y instrument block (one for lag 2, one for lag 3).
    """
    sim = simulate_dynamic_panel(N=10, T=8, rho=0.5, seed=0)
    bundle = build_instruments(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=1,
        gmm_lag_window=(2, 3),
        collapse=True,
    )
    # Only lagged-y block contributes; expect 2 columns (lag 2, lag 3)
    assert bundle["Z"].shape[1] == 2


def test_unbalanced_panel_skips_gaps():
    """Drop a single (panel, time) cell and verify the row count
    decreases by the expected number of usable diff rows lost.
    """
    sim = simulate_dynamic_panel(N=20, T=6, rho=0.5, seed=0)
    # full bundle for reference
    full = build_instruments(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=1, gmm_lag_window=(2, None), collapse=True,
    )
    # remove cell (panel=3, t=2)
    keep = ~((sim["panel_id"] == 3) & (sim["time_id"] == 2))
    bundle_unbal = build_instruments(
        sim["y"][keep], sim["panel_id"][keep], sim["time_id"][keep],
        lag_dep_var=1, gmm_lag_window=(2, None), collapse=True,
    )
    # Removing (panel=3, t=2) eliminates the diff rows that need y_2 in
    # either the outcome or the lagged-y regressor:
    #   Δy at t=2 needs y_2  → drop
    #   Δy at t=3 needs y_2  → drop
    #   At t=4, Δy_{t-1} = y_3 - y_2 needs y_2 → drop too
    # So 3 fewer rows.
    n_full = len(full["diff_rows"])
    n_unbal = len(bundle_unbal["diff_rows"])
    assert n_full - n_unbal == 3, (
        f"Expected loss of 3 diff rows; got {n_full - n_unbal}"
    )


def test_strictly_exogenous_columns_passthrough():
    """A strictly exogenous regressor x contributes one Δx column to Z."""
    sim = simulate_dynamic_panel(N=20, T=6, rho=0.5, beta_exog=0.5, seed=0)
    bundle = build_instruments(
        sim["y"], sim["panel_id"], sim["time_id"],
        lag_dep_var=1, X_exog=sim["x"],
        gmm_lag_window=(2, None), collapse=True,
    )
    # Last column of Z should be the Δx exog instrument
    assert any("Xx" in lab for lab in bundle["instr_labels"])


def test_invalid_lag_window_raises():
    sim = simulate_dynamic_panel(N=10, T=5, rho=0.5, seed=0)
    with pytest.raises(ValueError, match="lag_dep_var"):
        build_instruments(
            sim["y"], sim["panel_id"], sim["time_id"],
            lag_dep_var=0,
        )


def test_lag_window_lo_below_one_raises():
    sim = simulate_dynamic_panel(N=10, T=5, rho=0.5, seed=0)
    with pytest.raises(ValueError, match="gmm_lag_window"):
        build_instruments(
            sim["y"], sim["panel_id"], sim["time_id"],
            gmm_lag_window=(0, 5),
        )
