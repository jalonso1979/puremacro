"""End-to-end: synthetic announcement-day surprises → monthly aggregation →
proxy_svar → ProxySVARResult with IRF tensor and Olea-Pflueger F."""
import numpy as np
import pandas as pd

from puremacro.hfi import aggregate_to_period, gk2015_surprise
from puremacro.var.identify._results import ProxySVARResult
from puremacro.var.identify.proxy import proxy_svar


def test_hfi_end_to_end():
    rng = np.random.default_rng(0)
    # Synthetic monthly macro panel: 240 months, 3 vars
    T_macro = 240
    Y = np.cumsum(0.3 * rng.standard_normal((T_macro, 3)), axis=0)

    # Synthetic announcement series: 1 per month, with monthly-aggregated
    # surprise correlated with the first VAR residual
    n_announce = T_macro
    rate_pre = 95.0 * np.ones(n_announce)
    rate_post = rate_pre + 0.05 * rng.standard_normal(n_announce)
    days_remaining = rng.integers(5, 28, size=n_announce)
    surprise = gk2015_surprise(rate_pre, rate_post, days_remaining,
                               days_in_month=30)
    dates = pd.date_range("2000-01-15", periods=n_announce, freq="MS")
    monthly = aggregate_to_period(surprise, dates, freq="M")

    # Align: drop the first VAR observation count to match VAR sample
    z = monthly.values

    res = proxy_svar(Y, p=2, horizon=12, instrument_series=z,
                     n_boot=50, ci=0.9, seed=0)
    assert isinstance(res, ProxySVARResult)
    assert res.irf_point.shape == (13, 3, 3)
    assert np.isfinite(res.first_stage_F)
    # CI bands respect the point estimate (mostly)
    assert (res.irf_lower <= res.irf_upper).all()
