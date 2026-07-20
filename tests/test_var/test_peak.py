"""Tests for puremacro.var.peak — peak_summary (lifted) and peak_distribution (new)."""
import numpy as np
import pandas as pd
import pytest


def _toy_results(*, H=20, n=4):
    """Two-country toy results dict matching the structure produced by
    puremacro.var.identify.cholesky's per-country wrappers: each value is a
    dict with `point`, `lo`, `hi` arrays of shape (H, n, n) and `n_obs` int.

    Country A's IRF for shock 0 → response 2 has its peak at h=5 (positive 1.0
    in period 5, zero elsewhere); country B's at h=8 (negative 0.5).
    """
    def _zeros():
        return np.zeros((H, n, n))

    pt_a, lo_a, hi_a = _zeros(), _zeros(), _zeros()
    pt_a[5, 2, 0] = 1.0
    lo_a[5, 2, 0] = 0.5
    hi_a[5, 2, 0] = 1.5

    pt_b, lo_b, hi_b = _zeros(), _zeros(), _zeros()
    pt_b[8, 2, 0] = -0.5
    lo_b[8, 2, 0] = -0.9
    hi_b[8, 2, 0] = -0.1

    return {
        "AAA": {"point": pt_a, "lo": lo_a, "hi": hi_a, "n_obs": 100},
        "BBB": {"point": pt_b, "lo": lo_b, "hi": hi_b, "n_obs": 80},
    }


def test_peak_summary_lifted_finds_peak_horizon_and_value():
    from puremacro.var.peak import peak_summary

    df = peak_summary(_toy_results(), shock_idx=0, response_idx=2,
                      scale=100.0, cumulative=False)

    assert list(df.index) == ["AAA", "BBB"]
    assert df.loc["AAA", "peak_h"] == 5
    assert df.loc["BBB", "peak_h"] == 8
    # scale=100 so the values are in percent
    assert df.loc["AAA", "peak"] == pytest.approx(100.0)
    assert df.loc["BBB", "peak"] == pytest.approx(-50.0)
    # bootstrap bounds at the peak horizon
    assert df.loc["AAA", "peak_lo"] == pytest.approx(50.0)
    assert df.loc["AAA", "peak_hi"] == pytest.approx(150.0)
    # accum_h16 = cumulative sum at h=16 of the period IRF; for AAA that's
    # the single 1.0 at h=5, so cumulative 100.0 at h=16.
    assert df.loc["AAA", "accum_h16"] == pytest.approx(100.0)
    assert df.loc["BBB", "accum_h16"] == pytest.approx(-50.0)
    assert int(df.loc["AAA", "n_obs"]) == 100


def test_peak_summary_cumulative_finds_peak_of_cumsum():
    from puremacro.var.peak import peak_summary

    # Period IRFs: AAA has +1 at h=2 and +1 at h=10, so cumulative peaks at h=10
    # at value 2.0; BBB has -1 at h=3 then +0.5 at h=12, cumulative peak at h=3
    # (largest |·| of [-1.0, ..., -0.5]) at -1.0.
    H, n = 20, 4
    res = {
        "AAA": {"point": np.zeros((H, n, n)), "lo": np.zeros((H, n, n)),
                "hi": np.zeros((H, n, n)), "n_obs": 100},
        "BBB": {"point": np.zeros((H, n, n)), "lo": np.zeros((H, n, n)),
                "hi": np.zeros((H, n, n)), "n_obs": 80},
    }
    res["AAA"]["point"][2, 2, 0] = 1.0
    res["AAA"]["point"][10, 2, 0] = 1.0
    res["BBB"]["point"][3, 2, 0] = -1.0
    res["BBB"]["point"][12, 2, 0] = 0.5

    df = peak_summary(res, shock_idx=0, response_idx=2, scale=1.0, cumulative=True)
    assert df.loc["AAA", "peak_h"] == 10
    assert df.loc["AAA", "peak"] == pytest.approx(2.0)
    assert df.loc["BBB", "peak_h"] == 3
    assert df.loc["BBB", "peak"] == pytest.approx(-1.0)


def test_peak_distribution_projects_peak_summary():
    from puremacro.var.peak import peak_summary, peak_distribution

    res = _toy_results()
    full = peak_summary(res, shock_idx=0, response_idx=2,
                        scale=100.0, cumulative=False)
    proj = peak_distribution(res, shock_idx=0, response_idx=2,
                             scale=100.0, cumulative=False, h_fixed=16)

    assert list(proj.columns) == ["peak", "peak_h", "accum", "h_fixed", "n_obs"]
    assert (proj["h_fixed"] == 16).all()
    # peak / peak_h / accum / n_obs match peak_summary row-by-row
    # reset full's index for comparison since proj has RangeIndex
    full_reset = full.reset_index(drop=True)
    assert (proj["peak"] == full_reset["peak"]).all()
    assert (proj["peak_h"] == full_reset["peak_h"]).all()
    assert (proj["accum"] == full_reset["accum_h16"]).all()
    assert (proj["n_obs"] == full_reset["n_obs"]).all()


def test_peak_distribution_h_fixed_eight_uses_h8_not_h16():
    from puremacro.var.peak import peak_distribution

    H, n = 20, 4
    res = {"AAA": {"point": np.zeros((H, n, n)), "lo": np.zeros((H, n, n)),
                   "hi": np.zeros((H, n, n)), "n_obs": 100}}
    # Period IRF +1 at h=4 only; cumulative is 0 before h=4, then 1 thereafter.
    # accum at h=8 should be 1.0; accum at h=16 also 1.0 — but the column is
    # `accum`, not `accum_h16`, and h_fixed in the output column should be 8.
    res["AAA"]["point"][4, 2, 0] = 1.0

    proj = peak_distribution(res, shock_idx=0, response_idx=2,
                             scale=1.0, cumulative=True, h_fixed=8)
    assert proj.loc[0, "h_fixed"] == 8
    assert proj.loc[0, "accum"] == pytest.approx(1.0)


def test_peak_summary_handles_empty_results_dict():
    from puremacro.var.peak import peak_summary

    df = peak_summary({}, shock_idx=0, response_idx=0)
    assert df.empty


def test_peak_summary_short_horizon_nans():
    """When H <= 16 (or 8, or 4), the corresponding irf_* / accum_h16* fields
    must be NaN rather than crashing on out-of-bounds indexing.
    """
    from puremacro.var.peak import peak_summary

    # H = 12: irf_4 and irf_8 populated; irf_16 NaN; accum_h16 fields NaN.
    H, n = 12, 4
    res = {
        "AAA": {
            "point": np.zeros((H, n, n)),
            "lo":    np.zeros((H, n, n)),
            "hi":    np.zeros((H, n, n)),
            "n_obs": 50,
        }
    }
    res["AAA"]["point"][6, 2, 0] = 1.0  # so peak_h = 6, peak = 1.0
    df = peak_summary(res, shock_idx=0, response_idx=2, scale=1.0, cumulative=False)

    assert df.loc["AAA", "peak_h"] == 6
    assert df.loc["AAA", "peak"] == pytest.approx(1.0)
    assert df.loc["AAA", "irf_4"] == pytest.approx(0.0)
    assert df.loc["AAA", "irf_8"] == pytest.approx(0.0)
    assert pd.isna(df.loc["AAA", "irf_16"])
    assert pd.isna(df.loc["AAA", "accum_h16"])
    assert pd.isna(df.loc["AAA", "accum_h16_lo"])
    assert pd.isna(df.loc["AAA", "accum_h16_hi"])

    # H = 5: irf_4 populated, irf_8 and irf_16 NaN.
    H_short = 5
    res2 = {
        "BBB": {
            "point": np.zeros((H_short, n, n)),
            "lo":    np.zeros((H_short, n, n)),
            "hi":    np.zeros((H_short, n, n)),
            "n_obs": 30,
        }
    }
    res2["BBB"]["point"][2, 2, 0] = 0.5
    df2 = peak_summary(res2, shock_idx=0, response_idx=2, scale=1.0, cumulative=False)
    assert df2.loc["BBB", "irf_4"] == pytest.approx(0.0)
    assert pd.isna(df2.loc["BBB", "irf_8"])
    assert pd.isna(df2.loc["BBB", "irf_16"])
