"""Tests for puremacro.hfi.surprises."""
import numpy as np
import pandas as pd
import pytest

from puremacro.hfi.surprises import gk2015_surprise


def test_gk2015_surprise_no_scaling_at_month_start():
    """When announcement is on day 0 of the month, scaling factor is 1
    (M / (M - 0) = 1)."""
    pre = np.array([95.0])     # 100 - rate, so rate=5.0
    post = np.array([95.05])   # rate=4.95 → -5bp surprise (rate down)
    days_remaining = np.array([30])  # full month remaining
    s = gk2015_surprise(pre, post, days_remaining_in_month=days_remaining,
                        days_in_month=30)
    np.testing.assert_allclose(s, post - pre)


def test_gk2015_surprise_scaling_at_month_end():
    """Announcement near month end → scaling factor blows up (M / (M - d) large
    when d → M-1)."""
    pre = np.array([95.0])
    post = np.array([95.05])
    days_remaining = np.array([1])
    s = gk2015_surprise(pre, post, days_remaining_in_month=days_remaining,
                        days_in_month=30)
    # scale = 30 / 1 = 30
    np.testing.assert_allclose(s, (post - pre) * 30)


def test_gk2015_surprise_vector_inputs():
    """Multiple announcements aggregated correctly."""
    pre = np.array([95.0, 96.0, 97.0])
    post = np.array([95.10, 95.95, 97.05])
    days_remaining = np.array([15, 5, 25])
    s = gk2015_surprise(pre, post, days_remaining_in_month=days_remaining,
                        days_in_month=30)
    # raw changes: 0.10, -0.05, 0.05; scales: 30/15, 30/5, 30/25 = 2, 6, 1.2
    np.testing.assert_allclose(s, np.array([0.10 * 2, -0.05 * 6, 0.05 * 1.2]))


def test_gk2015_surprise_rejects_zero_remaining():
    """days_remaining=0 would imply announcement after month end — should error."""
    with pytest.raises(ValueError):
        gk2015_surprise(np.array([95.0]), np.array([95.1]),
                        days_remaining_in_month=np.array([0]),
                        days_in_month=30)


from puremacro.hfi.surprises import ns2018_first_pc


def test_ns2018_first_pc_recovers_dominant_factor():
    """If a single latent factor drives all K contracts, the first PC should
    recover it (up to sign) and explain near-100% of variance."""
    rng = np.random.default_rng(0)
    T, K = 200, 5
    factor = rng.standard_normal(T)
    loadings = rng.uniform(0.5, 1.5, K)
    surprise_matrix = np.outer(factor, loadings) + 0.01 * rng.standard_normal((T, K))
    pc, recovered_loadings = ns2018_first_pc(surprise_matrix, scale_to_idx=0)
    # Correlation with true factor should be near ±1
    corr = np.corrcoef(pc, factor)[0, 1]
    assert abs(corr) > 0.99


def test_ns2018_first_pc_scaling_to_target_contract():
    """When ``scale_to_idx=k``, a unit of the PC should correspond to ~1 unit
    of contract k (under perfect-correlation conditions)."""
    rng = np.random.default_rng(1)
    T, K = 300, 4
    factor = rng.standard_normal(T)
    loadings = np.array([1.0, 2.0, 0.5, 1.5])
    surprise_matrix = np.outer(factor, loadings)  # noiseless, perfect correlation
    pc, recovered_loadings = ns2018_first_pc(surprise_matrix, scale_to_idx=1)
    # The PC, when multiplied by the recovered loading on contract 1, should
    # recover contract 1's series (up to the column-mean removed by SVD demeaning).
    target_centered = surprise_matrix[:, 1] - surprise_matrix[:, 1].mean()
    np.testing.assert_allclose(
        pc * recovered_loadings[1], target_centered, atol=1e-8
    )


def test_ns2018_first_pc_orthogonal_to_residual():
    """The first-PC should be orthogonal to the residual signal (X - pc·loadings')."""
    rng = np.random.default_rng(2)
    surprise_matrix = rng.standard_normal((150, 4))
    pc, loadings = ns2018_first_pc(surprise_matrix, scale_to_idx=0)
    residual = surprise_matrix - np.outer(pc, loadings)
    np.testing.assert_allclose(pc @ residual, 0.0, atol=1e-8)


from puremacro.hfi.surprises import aggregate_to_period


def test_aggregate_to_period_monthly():
    """Sum announcements within each month."""
    surprises = np.array([0.10, -0.05, 0.20, 0.03])
    dates = pd.to_datetime(["2020-01-15", "2020-01-29", "2020-02-12", "2020-03-08"])
    out = aggregate_to_period(surprises, dates, freq="M")
    assert out.loc["2020-01"] == pytest.approx(0.05)
    assert out.loc["2020-02"] == pytest.approx(0.20)
    assert out.loc["2020-03"] == pytest.approx(0.03)


def test_aggregate_to_period_quarterly():
    surprises = np.array([1.0, 2.0, 3.0])
    dates = pd.to_datetime(["2020-01-15", "2020-02-12", "2020-04-08"])
    out = aggregate_to_period(surprises, dates, freq="Q")
    assert out.loc["2020Q1"] == pytest.approx(3.0)
    assert out.loc["2020Q2"] == pytest.approx(3.0)


def test_aggregate_to_period_fills_missing_with_zero():
    """Periods with no announcement should appear with value 0, not be dropped."""
    surprises = np.array([1.0, 2.0])
    dates = pd.to_datetime(["2020-01-15", "2020-04-08"])
    out = aggregate_to_period(surprises, dates, freq="M")
    # Feb and Mar 2020 should be present and zero
    assert "2020-02" in out.index.astype(str).tolist() or out.loc["2020-02"] == 0.0
    assert out.loc["2020-02"] == pytest.approx(0.0)
    assert out.loc["2020-03"] == pytest.approx(0.0)
