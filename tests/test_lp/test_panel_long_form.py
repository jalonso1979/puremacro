"""Regression tests: panel LP estimators accept long-form frames via
unit_col/time_col (docs/lp.md block 4, README one-liner) and panel_lp
exposes cov_type='driscoll-kraay' (docs/es/lp.md block 3).

Before the fix ``panel_lp_dk(panel_df, ..., unit_col='country',
time_col='date')`` raised ``TypeError: unexpected keyword argument
'unit_col'`` (C11) and ``panel_lp(..., cov_type='driscoll-kraay')`` raised
the same for ``cov_type`` (C12).
"""
import numpy as np
import pandas as pd
import pytest

from puremacro.lp import (
    cce_panel_lp,
    mean_group_panel_lp,
    panel_lp,
    panel_lp_dk,
    panel_lp_iv,
)


def _long_panel(seed: int = 0, N: int = 6, T: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        x = rng.standard_normal(T)
        z = 0.7 * x + 0.3 * rng.standard_normal(T)
        y = np.cumsum(0.5 * x + 0.3 * rng.standard_normal(T)) + i
        for t in range(T):
            rows.append({"country": f"C{i}", "date": pd.Period("2000Q1", freq="Q") + t,
                         "y": y[t], "x": x[t], "z": z[t]})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("fn, kw", [
    (panel_lp, {}),
    (panel_lp_dk, {}),
    (panel_lp_iv, {"z": "z"}),
    (cce_panel_lp, {}),
    (mean_group_panel_lp, {"min_obs": 20}),
])
def test_unit_col_time_col_equals_multiindex_call(fn, kw):
    long = _long_panel()
    wide = long.set_index(["country", "date"])
    a = fn(long, y="y", x="x", horizon=3, lags=1, unit_col="country", time_col="date", **kw)
    b = fn(wide, y="y", x="x", horizon=3, lags=1, entity_level="country", time_level="date", **kw)
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))
    # unit_col/time_col may also name the levels of an existing MultiIndex
    c = fn(wide, y="y", x="x", horizon=3, lags=1, unit_col="country", time_col="date", **kw)
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(c))


def test_shuffled_long_frame_is_row_order_invariant():
    long = _long_panel()
    a = panel_lp_dk(long, y="y", x="x", horizon=2, lags=1, unit_col="country", time_col="date")
    b = panel_lp_dk(long.sample(frac=1.0, random_state=1), y="y", x="x", horizon=2, lags=1,
                    unit_col="country", time_col="date")
    np.testing.assert_allclose(a.values.astype(float), b.values.astype(float))


def test_entity_level_time_level_accept_long_frame_columns():
    """A long frame whose identifier columns carry the level names is
    indexed automatically instead of failing inside two_way_fe_within."""
    long = _long_panel()
    a = panel_lp(long, y="y", x="x", horizon=2, lags=1, entity_level="country", time_level="date")
    b = panel_lp(long.set_index(["country", "date"]), y="y", x="x", horizon=2, lags=1,
                 entity_level="country", time_level="date")
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))


def test_reversed_multiindex_levels_are_reordered():
    long = _long_panel()
    wide = long.set_index(["country", "date"])
    swapped = long.set_index(["date", "country"])
    a = panel_lp(wide, y="y", x="x", horizon=2, lags=1, entity_level="country", time_level="date")
    b = panel_lp(swapped, y="y", x="x", horizon=2, lags=1, entity_level="country", time_level="date")
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(b))
    c = panel_lp(swapped, y="y", x="x", horizon=2, lags=1, unit_col="country", time_col="date")
    pd.testing.assert_frame_equal(pd.DataFrame(a), pd.DataFrame(c))


def test_cov_type_driscoll_kraay_matches_panel_lp_dk():
    long = _long_panel()
    dk = panel_lp_dk(long, y="y", x="x", horizon=4, lags=1, unit_col="country", time_col="date")
    via = panel_lp(long, y="y", x="x", horizon=4, lags=1, unit_col="country", time_col="date",
                   cov_type="driscoll-kraay")
    pd.testing.assert_frame_equal(pd.DataFrame(dk), pd.DataFrame(via))
    cl = panel_lp(long, y="y", x="x", horizon=4, lags=1, unit_col="country", time_col="date")
    np.testing.assert_allclose(cl.point, dk.point)
    assert not np.allclose(cl.se, dk.se)
    alias = panel_lp(long, y="y", x="x", horizon=4, lags=1, unit_col="country", time_col="date",
                     cov_type="dk")
    pd.testing.assert_frame_equal(pd.DataFrame(dk), pd.DataFrame(alias))
    with pytest.raises(ValueError, match="cov_type"):
        panel_lp(long, y="y", x="x", horizon=1, unit_col="country", time_col="date",
                 cov_type="robust")


def test_long_frame_errors_are_explicit():
    long = _long_panel()
    with pytest.raises(ValueError, match="unit_col"):
        panel_lp_dk(long, y="y", x="x", horizon=1, unit_col="nope", time_col="date")
    with pytest.raises(ValueError, match="both"):
        panel_lp_dk(long, y="y", x="x", horizon=1, unit_col="country")
    with pytest.raises(ValueError, match="MultiIndex"):
        panel_lp_dk(long, y="y", x="x", horizon=1)          # default level names absent
